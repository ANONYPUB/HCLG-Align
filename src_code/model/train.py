import json
import logging
import os
import random
from typing import Any, Dict, Optional

from config import (
    LR, LR_GNN, WEIGHT_DECAY, MAX_GRAD_NORM, GRAD_ACCUM_STEPS,
    NUM_EPOCHS, WARMUP_RATIO,
    LAMBDA_INIT, LAMBDA_FINAL, LAMBDA_PHASE1_RATIO,
    train_log_path, model_save_dir, best_ckpt_dir, latest_ckpt_dir,
    BACKBONE_MODEL_NAME, TRUST_REMOTE_CODE, HF_CACHE_DIR,
    train_raw_hclg_path, valid_raw_hclg_path,
    TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE,
    val_pred_path,
    seed_torch, seeds,
    EVAL_ROUGE_START_EPOCH,
    EVAL_LANGUAGE,
)

import numpy as np
import torch
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup, AutoTokenizer

from accelerate import Accelerator
from accelerate.utils import gather_object

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_IMPORT_ERR: Optional[str] = None
except Exception as _tb_e:
    SummaryWriter = None
    _TB_IMPORT_ERR = repr(_tb_e)

from data import build_dataloader
from model import HCLGAlign
from evaluate import compute_rouge



def setup_logging(log_path: str) -> logging.Logger:
    """配置文件 + 控制台双输出 logger.

    config.py 已经 basicConfig 了一个控制台 handler, 这里只是额外加一个
    FileHandler 写到 log_path, 让训练日志持久化.

    Args:
        log_path: 训练日志文件路径 (一般是 config.train_log_path)
    Returns:
        配好的 root logger (后续用 logging.info(...) 即可)
    """

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s : %(levelname)s : %(message)s'))
    fh.setLevel(logging.INFO)

    root = logging.getLogger()
    has_file = any(isinstance(h, logging.FileHandler) and
                    os.path.abspath(h.baseFilename) == os.path.abspath(log_path)
                    for h in root.handlers)
    if not has_file:
        root.addHandler(fh)

    return root


def get_lambda(epoch: int,
               total_epochs: int = NUM_EPOCHS,
               lambda_init: float = LAMBDA_INIT,
               lambda_final: float = LAMBDA_FINAL,
               phase1_ratio: float = LAMBDA_PHASE1_RATIO) -> float:

    phase1_end = int(total_epochs * phase1_ratio)

    if epoch < phase1_end:
        return lambda_init

    progress = (epoch - phase1_end) / max(1, total_epochs - phase1_end - 1)
    progress = min(1.0, progress)
    return lambda_init + progress * (lambda_final - lambda_init)


def build_optimizer(model: torch.nn.Module,
                    lora_lr: float = LR,
                    gnn_lr: float = LR_GNN,
                    weight_decay: float = WEIGHT_DECAY) -> torch.optim.Optimizer:

    lora_params = []
    gnn_params  = []
    frozen_params_count = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            frozen_params_count += 1
            continue
        if 'lora_' in name:
            lora_params.append(param)
        else:
            gnn_params.append(param)

    n_lora = sum(p.numel() for p in lora_params)
    n_gnn  = sum(p.numel() for p in gnn_params)
    logging.info('Optimizer param groups:')
    logging.info('  LoRA params  : %d 个张量, %.2fM 参数, lr=%g',
                  len(lora_params), n_lora / 1e6, lora_lr)
    logging.info('  GNN  params  : %d 个张量, %.2fM 参数, lr=%g',
                  len(gnn_params), n_gnn / 1e6, gnn_lr)
    logging.info('  Frozen params: %d 个张量 (backbone 主体)', frozen_params_count)

    return torch.optim.AdamW([
        {'params': lora_params, 'lr': lora_lr, 'name': 'lora'},
        {'params': gnn_params,  'lr': gnn_lr,  'name': 'gnn'},
    ], weight_decay=weight_decay)


def build_lr_scheduler(optimizer: torch.optim.Optimizer,
                       num_training_steps: int,
                       warmup_ratio: float = WARMUP_RATIO):

    warmup_steps = int(num_training_steps * warmup_ratio)
    logging.info('LR scheduler: warmup=%d steps, total=%d steps (warmup_ratio=%.2f)',
                  warmup_steps, num_training_steps, warmup_ratio)
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )




def _save_rng_state() -> Dict[str, Any]:

    return {
        'python_random':  random.getstate(),
        'numpy_random':   np.random.get_state(),
        'torch_cpu':      torch.get_rng_state(),
        'torch_cuda':     torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _load_rng_state(state: Dict[str, Any]) -> None:

    random.setstate(state['python_random'])
    np.random.set_state(state['numpy_random'])
    torch.set_rng_state(state['torch_cpu'])
    if torch.cuda.is_available() and state.get('torch_cuda') is not None:
        torch.cuda.set_rng_state_all(state['torch_cuda'])


def save_checkpoint(ckpt_dir: str,
                     model,
                     optimizer: Optional[torch.optim.Optimizer] = None,
                     scheduler = None,
                     accelerator = None,
                     meta: Optional[Dict[str, Any]] = None,
                     save_optimizer: bool = False) -> None:

    if accelerator is not None and not accelerator.is_main_process:
        return

    os.makedirs(ckpt_dir, exist_ok=True)

    raw_model = model
    if accelerator is not None:
        raw_model = accelerator.unwrap_model(model)

    lora_dir = os.path.join(ckpt_dir, 'lora')
    raw_model.backbone.save_pretrained(lora_dir)

    gnn_state = {
        'node_init':       raw_model.node_init.state_dict(),
        'rgcn':            raw_model.rgcn.state_dict(),
        'proj':            raw_model.proj.state_dict(),
        'readout':         raw_model.readout.state_dict(),
        'log_logit_scale': raw_model.log_logit_scale.detach().cpu(),
    }
    if getattr(raw_model, 'layer_injector', None) is not None:
        gnn_state['layer_injector'] = raw_model.layer_injector.state_dict()
    if getattr(raw_model, 'cross_attn', None) is not None:
        gnn_state['cross_attn'] = raw_model.cross_attn.state_dict()
    torch.save(gnn_state, os.path.join(ckpt_dir, 'gnn.pt'))

    if save_optimizer and optimizer is not None:
        opt_state = {
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
            'rng':       _save_rng_state(),
        }
        torch.save(opt_state, os.path.join(ckpt_dir, 'optimizer.pt'))

    meta = meta or {}
    with open(os.path.join(ckpt_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logging.info('Checkpoint saved to %s (save_optimizer=%s)',
                  ckpt_dir, save_optimizer)


def load_checkpoint(ckpt_dir: str,
                     model,
                     optimizer: Optional[torch.optim.Optimizer] = None,
                     scheduler = None) -> Dict[str, Any]:

    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f'Checkpoint dir not found: {ckpt_dir}')

    logging.info('Loading checkpoint from %s', ckpt_dir)

    from peft import set_peft_model_state_dict
    lora_dir = os.path.join(ckpt_dir, 'lora')
    safetensors_path = os.path.join(lora_dir, 'adapter_model.safetensors')
    bin_path         = os.path.join(lora_dir, 'adapter_model.bin')
    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        adapter_state = load_file(safetensors_path)
    elif os.path.exists(bin_path):
        adapter_state = torch.load(bin_path, map_location='cpu')
    else:
        raise FileNotFoundError(f'No adapter weights found in {lora_dir}')
    load_result = set_peft_model_state_dict(model.backbone, adapter_state)
    if hasattr(load_result, 'missing_keys') and (load_result.missing_keys or load_result.unexpected_keys):
        logging.warning('PEFT load: missing_keys=%d, unexpected_keys=%d (LoRA 配置可能改过)',
                         len(load_result.missing_keys), len(load_result.unexpected_keys))

    gnn_path = os.path.join(ckpt_dir, 'gnn.pt')
    gnn_state = torch.load(gnn_path, map_location='cpu')
    model.node_init.load_state_dict(gnn_state['node_init'])
    model.rgcn.load_state_dict(gnn_state['rgcn'])
    model.proj.load_state_dict(gnn_state['proj'])
    if 'readout' in gnn_state:
        model.readout.load_state_dict(gnn_state['readout'])
    else:
        logging.info('  ckpt 不含 readout (老格式), AttentionPool 用默认初始化 (= mean pool)')
    model.log_logit_scale.data.copy_(gnn_state['log_logit_scale'])

    if 'layer_injector' in gnn_state and getattr(model, 'layer_injector', None) is not None:
        model.layer_injector.load_state_dict(gnn_state['layer_injector'])
        logging.info('  ✓ layer_injector state 已加载 (inject hook 路径)')
    elif getattr(model, 'layer_injector', None) is not None:
        logging.info('  ckpt 不含 layer_injector (老格式), 用初始化权重 (tanh(0)=0, 等价 pure-LLM 起步)')
    elif 'layer_injector' in gnn_state:
        logging.warning('  ckpt 含 layer_injector 但当前模型未启用 injector, 跳过')

    if 'cross_attn' in gnn_state and getattr(model, 'cross_attn', None) is not None:
        model.cross_attn.load_state_dict(gnn_state['cross_attn'])
        logging.info('  ✓ cross_attn state 已加载 (v9 cross-attn 路径)')
    elif getattr(model, 'cross_attn', None) is not None:
        logging.info('  ckpt 不含 cross_attn (老格式), 用初始化权重 (gate=0, 等价 pure-LLM 起步)')
    elif 'cross_attn' in gnn_state:
        logging.warning('  ckpt 含 cross_attn 但当前模型未启用 crossattn, 跳过')

    opt_path = os.path.join(ckpt_dir, 'optimizer.pt')
    if os.path.exists(opt_path):
        opt_state = torch.load(opt_path, map_location='cpu', weights_only=False)
        if optimizer is not None:
            optimizer.load_state_dict(opt_state['optimizer'])
            logging.info('  optimizer state loaded')
        if scheduler is not None and opt_state.get('scheduler') is not None:
            scheduler.load_state_dict(opt_state['scheduler'])
            logging.info('  scheduler state loaded')
        if 'rng' in opt_state:
            _load_rng_state(opt_state['rng'])
            logging.info('  RNG state loaded')
    else:
        logging.info('  optimizer.pt not found (probably loading from best/), '
                      'optimizer / scheduler / RNG NOT restored')

    meta_path = os.path.join(ckpt_dir, 'meta.json')
    if os.path.exists(meta_path):
        with open(meta_path, encoding='utf-8') as f:
            meta = json.load(f)
    else:
        meta = {}

    logging.info('Checkpoint loaded. Resume from epoch=%d, best_val_rouge_l=%.4f',
                  meta.get('epoch', -1), meta.get('best_val_rouge_l', 0.0))
    return meta




def train_one_epoch(accelerator: Accelerator,
                     model,
                     train_loader,
                     optimizer: torch.optim.Optimizer,
                     scheduler,
                     lambda_cl: float,
                     epoch: int,
                     total_epochs: int = NUM_EPOCHS,
                     tb_writer=None,
                     global_step_start: int = 0,
                     max_train_batches: int = 0) -> Dict[str, float]:

    model.train()

    sum_L_gen, sum_L_cl, sum_loss, n_steps = 0.0, 0.0, 0.0, 0

    pbar = tqdm(train_loader,
                 desc=f'Epoch {epoch+1:02d}/{total_epochs} train',
                 disable=not accelerator.is_main_process,
                 dynamic_ncols=True)

    for step, batch in enumerate(pbar):
        with accelerator.accumulate(model):
            outputs = model(batch)
            L_gen = outputs['L_gen']
            L_cl  = outputs['L_cl']
            loss = L_gen + lambda_cl * L_cl

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        sum_L_gen += L_gen.item()
        sum_L_cl  += L_cl.item()
        sum_loss  += loss.item()
        n_steps   += 1

        if accelerator.is_main_process and step % 10 == 0:
            pbar.set_postfix({
                'L_gen': f'{L_gen.item():.4f}',
                'L_cl':  f'{L_cl.item():.4f}',
                'λ':     f'{lambda_cl:.2f}',
                'lr':    f'{scheduler.get_last_lr()[0]:.2e}',
                'τ':     f'{outputs["tau"].item():.4f}',
            })

        if tb_writer is not None and accelerator.is_main_process:
            gs = global_step_start + step
            tb_writer.add_scalar('train/L_gen',  L_gen.item(),                gs)
            tb_writer.add_scalar('train/L_cl',   L_cl.item(),                 gs)
            tb_writer.add_scalar('train/loss',   loss.item(),                 gs)
            tb_writer.add_scalar('train/lambda', lambda_cl,                   gs)
            tb_writer.add_scalar('train/lr',     scheduler.get_last_lr()[0],  gs)
            tb_writer.add_scalar('train/tau',    outputs['tau'].item(),       gs)

        if max_train_batches and (step + 1) >= max_train_batches:
            if accelerator.is_main_process:
                logging.info('[debug] max_train_batches=%d 已达, 提前结束本 epoch', max_train_batches)
            break

    return {
        'train_L_gen': sum_L_gen / max(1, n_steps),
        'train_L_cl':  sum_L_cl / max(1, n_steps),
        'train_loss':  sum_loss / max(1, n_steps),
    }


@torch.no_grad()
def evaluate_val_loss(accelerator: Accelerator,
                       model,
                       val_loader) -> Dict[str, float]:

    model.eval()

    losses_per_step = []
    for batch in tqdm(val_loader, desc='Val L_gen',
                      disable=not accelerator.is_main_process,
                      dynamic_ncols=True):
        outputs = model(batch)
        l_gen_gathered = accelerator.gather(outputs['L_gen'].unsqueeze(0))
        losses_per_step.append(l_gen_gathered)

    if not losses_per_step:
        return {'val_L_gen': float('inf')}
    all_losses = torch.cat(losses_per_step)
    return {'val_L_gen': all_losses.mean().item()}


@torch.no_grad()
def evaluate_val_rouge(accelerator: Accelerator,
                        model,
                        tokenizer,
                        val_loader,
                        language: str = 'zh',
                        save_predictions_path: Optional[str] = None
                        ) -> Dict[str, float]:

    model.eval()
    raw_model = accelerator.unwrap_model(model)

    local_preds: List[str] = []
    local_refs:  List[str] = []
    local_ids:   List[Any] = []

    for batch in tqdm(val_loader, desc='Val ROUGE-L gen',
                       disable=not accelerator.is_main_process,
                       dynamic_ncols=True):
        texts = raw_model.generate(batch, tokenizer=tokenizer)
        local_preds.extend(texts)
        local_refs.extend(batch['impression_text'])
        local_ids.extend(batch['ids'])

    all_preds = gather_object(local_preds)
    all_refs  = gather_object(local_refs)
    all_ids   = gather_object(local_ids)

    rouge_result = {'rouge_l_p': 0.0, 'rouge_l_r': 0.0,
                     'rouge_l_f': 0.0, 'n_samples': 0}

    if accelerator.is_main_process:
        seen_ids = set()
        dedup_preds: List[str] = []
        dedup_refs:  List[str] = []
        dedup_ids:   List[Any] = []
        for sid, pred, ref in zip(all_ids, all_preds, all_refs):
            key = str(sid)
            if key in seen_ids:
                continue
            seen_ids.add(key)
            dedup_preds.append(pred)
            dedup_refs.append(ref)
            dedup_ids.append(sid)
        if len(dedup_preds) != len(all_preds):
            logging.info('Val dedup: %d -> %d samples (removed %d duplicates from DDP padding)',
                          len(all_preds), len(dedup_preds), len(all_preds) - len(dedup_preds))
        all_preds, all_refs, all_ids = dedup_preds, dedup_refs, dedup_ids

        rouge_result = compute_rouge(all_preds, all_refs, language=language)
        logging.info(
            'Val ROUGE-L: F1=%.4f, P=%.4f, R=%.4f (n=%d, lang=%s)',
            rouge_result['rouge_l_f'], rouge_result['rouge_l_p'],
            rouge_result['rouge_l_r'], rouge_result['n_samples'], language,
        )
        if save_predictions_path is not None:
            os.makedirs(os.path.dirname(save_predictions_path), exist_ok=True)
            with open(save_predictions_path, 'w', encoding='utf-8') as f:
                for sid, ref, pred in zip(all_ids, all_refs, all_preds):
                    f.write(json.dumps({
                        'id': sid, 'reference': ref, 'prediction': pred,
                    }, ensure_ascii=False) + '\n')

    rouge_l_f_t = torch.tensor(rouge_result['rouge_l_f'], device=accelerator.device)
    rouge_l_f_t = accelerator.reduce(rouge_l_f_t, reduction='sum')
    rouge_result['rouge_l_f'] = float(rouge_l_f_t.item())

    return rouge_result



def main(args) -> None:
    from accelerate.utils import DistributedDataParallelKwargs
    _ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        kwargs_handlers=[_ddp_kwargs],
    )
    seed_torch(seeds[1] + accelerator.process_index)

    if accelerator.is_main_process:
        os.makedirs(model_save_dir, exist_ok=True)
        setup_logging(train_log_path)
        logging.info('=' * 70)
        logging.info('HCLG-Align Training')
        logging.info('=' * 70)
        logging.info('Backbone     : %s', BACKBONE_MODEL_NAME)
        logging.info('Save dir     : %s', model_save_dir)
        logging.info('Num GPUs     : %d', accelerator.num_processes)
        logging.info('Grad accum   : %d', GRAD_ACCUM_STEPS)
        logging.info('Effective BS : %d',
                      TRAIN_BATCH_SIZE * GRAD_ACCUM_STEPS * accelerator.num_processes)
        logging.info('Mixed prec   : %s', accelerator.mixed_precision)
        logging.info('Resume       : %s', args.resume)
        logging.info('Language     : %s', args.language)

    tb_writer = None
    if accelerator.is_main_process:
        if SummaryWriter is None:
            logging.warning('tensorboard 不可用 (%s), 训练曲线不会被记录. '
                            'pip install tensorboard 即可启用.', _TB_IMPORT_ERR)
        else:
            tb_dir = os.path.join(model_save_dir, 'tb')
            os.makedirs(tb_dir, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=tb_dir)
            logging.info('TB log dir   : %s', tb_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        BACKBONE_MODEL_NAME,
        trust_remote_code=TRUST_REMOTE_CODE,
        cache_dir=HF_CACHE_DIR,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_loader = build_dataloader(
        train_raw_hclg_path, tokenizer,
        mode='train', batch_size=TRAIN_BATCH_SIZE,
    )
    val_loader = build_dataloader(
        valid_raw_hclg_path, tokenizer,
        mode='val', batch_size=EVAL_BATCH_SIZE,
    )
    if accelerator.is_main_process:
        logging.info('train_loader: %d batches, val_loader: %d batches',
                      len(train_loader), len(val_loader))

    if accelerator.is_main_process:
        logging.info('Instantiating HCLGAlign... (loading backbone, this may download weights)')
    model = HCLGAlign()

    optimizer = build_optimizer(model)
    steps_per_epoch = (len(train_loader) // accelerator.num_processes) // GRAD_ACCUM_STEPS
    num_training_steps = steps_per_epoch * NUM_EPOCHS
    scheduler = build_lr_scheduler(optimizer, num_training_steps)

    start_epoch = 0
    best_metric = -1.0
    if args.resume:
        if not os.path.exists(latest_ckpt_dir):
            if accelerator.is_main_process:
                logging.warning(
                    '--resume specified but %s does not exist. Starting from scratch.',
                    latest_ckpt_dir,
                )
        else:
            meta = load_checkpoint(latest_ckpt_dir, model, optimizer, scheduler)
            start_epoch = meta.get('epoch', -1) + 1
            best_metric = meta.get('best_val_rouge_l', -1.0)
            if accelerator.is_main_process:
                logging.info('Resumed: start_epoch=%d, best_val_rouge_l=%.4f',
                              start_epoch, best_metric)

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    for epoch in range(start_epoch, NUM_EPOCHS):
        lambda_cl = get_lambda(epoch, total_epochs=NUM_EPOCHS)

        if hasattr(train_loader, 'set_epoch'):
            train_loader.set_epoch(epoch)
        if hasattr(getattr(train_loader, 'sampler', None), 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        if hasattr(getattr(getattr(train_loader, 'batch_sampler', None), 'sampler', None), 'set_epoch'):
            train_loader.batch_sampler.sampler.set_epoch(epoch)

        if accelerator.is_main_process:
            logging.info('-' * 70)
            logging.info('Epoch %d/%d  λ=%.4f', epoch+1, NUM_EPOCHS, lambda_cl)

        train_metrics = train_one_epoch(
            accelerator, model, train_loader, optimizer, scheduler,
            lambda_cl=lambda_cl, epoch=epoch, total_epochs=NUM_EPOCHS,
            tb_writer=tb_writer,
            global_step_start=epoch * len(train_loader),
            max_train_batches=getattr(args, 'max_train_batches', 0),
        )

        val_loss_metrics = evaluate_val_loss(accelerator, model, val_loader)

        if (epoch + 1) >= EVAL_ROUGE_START_EPOCH:
            val_rouge_metrics = evaluate_val_rouge(
                accelerator, model, tokenizer, val_loader,
                language=args.language,
                save_predictions_path=val_pred_path if accelerator.is_main_process else None,
            )
        else:
            val_rouge_metrics = None
            if accelerator.is_main_process:
                logging.info('  (skip ROUGE-L eval: epoch %d < EVAL_ROUGE_START_EPOCH=%d)',
                             epoch + 1, EVAL_ROUGE_START_EPOCH)

        accelerator.wait_for_everyone()

        if accelerator.is_main_process and tb_writer is not None:
            tb_writer.add_scalar('epoch/train_L_gen', train_metrics['train_L_gen'],  epoch)
            tb_writer.add_scalar('epoch/train_L_cl',  train_metrics['train_L_cl'],   epoch)
            tb_writer.add_scalar('epoch/train_loss',  train_metrics['train_loss'],   epoch)
            tb_writer.add_scalar('epoch/val_L_gen',   val_loss_metrics['val_L_gen'], epoch)
            tb_writer.add_scalar('epoch/lambda',      lambda_cl,                     epoch)
            if val_rouge_metrics is not None:
                tb_writer.add_scalar('epoch/val_rouge_l_f', val_rouge_metrics['rouge_l_f'], epoch)
                tb_writer.add_scalar('epoch/val_rouge_l_p', val_rouge_metrics['rouge_l_p'], epoch)
                tb_writer.add_scalar('epoch/val_rouge_l_r', val_rouge_metrics['rouge_l_r'], epoch)
            tb_writer.flush()

        if accelerator.is_main_process:
            if val_rouge_metrics is not None:
                logging.info(
                    'Epoch %d done. train_loss=%.4f, val_L_gen=%.4f, '
                    'val_ROUGE-L_F1=%.4f (best so far=%.4f)',
                    epoch+1,
                    train_metrics['train_loss'],
                    val_loss_metrics['val_L_gen'],
                    val_rouge_metrics['rouge_l_f'],
                    max(best_metric, val_rouge_metrics['rouge_l_f']),
                )
            else:
                logging.info(
                    'Epoch %d done. train_loss=%.4f, val_L_gen=%.4f '
                    '(ROUGE-L skipped; best so far=%.4f)',
                    epoch+1,
                    train_metrics['train_loss'],
                    val_loss_metrics['val_L_gen'],
                    best_metric,
                )

        meta = {
            'epoch': epoch,
            'lambda': lambda_cl,
            'train_loss': train_metrics['train_loss'],
            'train_L_gen': train_metrics['train_L_gen'],
            'train_L_cl': train_metrics['train_L_cl'],
            'val_L_gen': val_loss_metrics['val_L_gen'],
            'best_val_rouge_l': best_metric,
        }
        if val_rouge_metrics is not None:
            meta['val_rouge_l_f'] = val_rouge_metrics['rouge_l_f']
            meta['val_rouge_l_p'] = val_rouge_metrics['rouge_l_p']
            meta['val_rouge_l_r'] = val_rouge_metrics['rouge_l_r']
            meta['best_val_rouge_l'] = max(best_metric, val_rouge_metrics['rouge_l_f'])
        save_checkpoint(
            latest_ckpt_dir, model, optimizer, scheduler,
            accelerator=accelerator, meta=meta, save_optimizer=True,
        )

        if val_rouge_metrics is not None and val_rouge_metrics['rouge_l_f'] > best_metric:
            best_metric = val_rouge_metrics['rouge_l_f']
            best_meta = dict(meta)
            best_meta['best_val_rouge_l'] = best_metric
            save_checkpoint(
                best_ckpt_dir, model, accelerator=accelerator,
                meta=best_meta, save_optimizer=False,
            )
            if accelerator.is_main_process:
                logging.info('  ★ New best ROUGE-L = %.4f at epoch %d', best_metric, epoch+1)

        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        logging.info('=' * 70)
        logging.info('Training complete. Best val ROUGE-L = %.4f', best_metric)
        logging.info('Best ckpt : %s', best_ckpt_dir)
        logging.info('Latest ckpt: %s', latest_ckpt_dir)
        if tb_writer is not None:
            tb_writer.close()
            logging.info('TB log dir : %s  (tensorboard --logdir %s)',
                         os.path.join(model_save_dir, 'tb'), os.path.join(model_save_dir, 'tb'))
        logging.info('=' * 70)



def run_selftest():
    print('=' * 70)
    print('Phase 1 self-test: 工具函数 + checkpoint I/O')
    print('=' * 70)

    print('\n[Test 1] λ 调度 (NUM_EPOCHS=50, phase1_ratio=0.3)')
    print('  epoch | λ')
    print('  ' + '-' * 25)
    for ep in [0, 5, 10, 14, 15, 20, 30, 40, 49]:
        lam = get_lambda(ep, total_epochs=50)
        bar = '#' * int(lam * 30)
        print(f'  {ep:5d} | {lam:.3f}  {bar}')

    print('\n[Test 2] setup_logging')
    test_log = '/tmp/hclg_align_test.log'
    setup_logging(test_log)
    logging.info('Hello from train.py self-test')
    print(f'  Log written to: {test_log}')
    print(f'  Tail of log file:')
    if os.path.exists(test_log):
        with open(test_log) as f:
            for line in f.readlines()[-3:]:
                print(f'    {line.rstrip()}')

    print('\n[Test 3] dummy 模型 save/load 往返')
    import tempfile
    import torch.nn as nn
    from torch_geometric.utils import scatter as _

    class _FakeBackbone:
        def save_pretrained(self, path):
            os.makedirs(path, exist_ok=True)
            torch.save({'fake_lora_A': torch.randn(4, 8)},
                        os.path.join(path, 'adapter_model.bin'))
            with open(os.path.join(path, 'adapter_config.json'), 'w') as f:
                json.dump({'r': 16, 'lora_alpha': 32}, f)

    class _FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.node_init = nn.Linear(4, 4)
            self.rgcn = nn.Linear(4, 4)
            self.proj = nn.Linear(4, 4)
            self.log_logit_scale = nn.Parameter(torch.tensor(2.66))
            self.backbone = _FakeBackbone()

    fake_model = _FakeModel()
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt = os.path.join(tmp_dir, 'test_ckpt')
        save_checkpoint(ckpt, fake_model, meta={'epoch': 7, 'note': 'dummy'},
                         save_optimizer=False)
        files = os.listdir(ckpt)
        print(f'  Saved files: {sorted(files)}')
        assert 'gnn.pt' in files
        assert 'meta.json' in files
        assert 'lora' in files
        with open(os.path.join(ckpt, 'meta.json'), encoding='utf-8') as f:
            saved_meta = json.load(f)
        assert saved_meta['epoch'] == 7
        gnn_state = torch.load(os.path.join(ckpt, 'gnn.pt'))
        assert 'node_init' in gnn_state
        assert 'log_logit_scale' in gnn_state
        print(f'  gnn.pt 包含: {list(gnn_state.keys())}')
        print(f'  meta.json: {saved_meta}')
    print('  ✓ save/load 往返一致')

    print('\n[Test 4] RNG state 保存 + 恢复')
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    rng_state = _save_rng_state()
    a = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    _load_rng_state(rng_state)
    b = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    print(f'  before restore: {a}')
    print(f'  after  restore: {b}')
    assert a == b, 'RNG state restore failed!'
    print('  ✓ RNG state 完美恢复')

    print('\n' + '=' * 70)
    print('阶段 1 self-test 全部通过 ✓')
    print('=' * 70)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='HCLG-Align 训练入口',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--selftest', action='store_true',
        help='跑阶段 1 self-test (λ 调度 + ckpt I/O), 无需 GPU',
    )
    parser.add_argument(
        '--resume', action='store_true',
        help=f'从 {latest_ckpt_dir} 续训, 恢复 epoch / optimizer / scheduler / RNG state',
    )
    parser.add_argument(
        '--language', default=EVAL_LANGUAGE, choices=['zh', 'en'],
        help='数据集语种 (zh=FUS 中文, en=MIMIC-CXR 英文), 影响 ROUGE-L 分词器选择. '
             '默认走 config.EVAL_LANGUAGE (跟随 config.DATASET 自动切换)',
    )
    parser.add_argument(
        '--max-train-batches', type=int, default=0, dest='max_train_batches',
        help='调试用: 每个 epoch 只跑前 N 个 batch 就停 (0 = 不限制). '
             '例: --max-train-batches 5 跑 5 个 batch 看不崩, 不浪费完整 epoch 时间.',
    )
    args = parser.parse_args()

    if args.selftest:
        run_selftest()
    else:
        main(args)
