import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from config import (
    BACKBONE_MODEL_NAME, TRUST_REMOTE_CODE, HF_CACHE_DIR,
    test_raw_hclg_path, train_raw_hclg_path, best_ckpt_dir,
    test_pred_path, eval_metrics_path, result_save_dir,
    EVAL_BATCH_SIZE,
    seed_torch, seeds,
    EVAL_LANGUAGE,
    TTC_K, TTC_SAMPLING_CONFIGS, TTC_LENGTH_WEIGHT, TTC_LENGTH_SIGMA,
    TTC_TARGET_LENGTH_AUTO, TTC_TARGET_LENGTH_FALLBACK,
    TTC_MBR_WEIGHT,
)

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.utils import gather_object

from data import build_dataloader
from model import HCLGAlign
from hclg_verifier import (
    HCLGVerifier,
    MBRScorer,
    load_test_hclg_entities,
    compute_target_length_from_jsonl,
)
from evaluate import (
    compute_rouge,
    compute_bertscore,
    compute_f1_chexbert,
    compute_f1_radgraph,
    compute_lsc_f1,
)
from train import load_checkpoint, setup_logging



@torch.no_grad()
def _generate_ttc_batch(raw_model,
                          batch: Dict[str, Any],
                          tokenizer,
                          verifier: HCLGVerifier,
                          hclg_dict: Dict[str, List[str]],
                          K: int) -> List[str]:

    B = len(batch['ids'])
    candidates_per_sample: List[List[str]] = [[] for _ in range(B)]

    n_configs = len(TTC_SAMPLING_CONFIGS)
    for round_idx in range((K + n_configs - 1) // n_configs):
        round_size = min(n_configs, K - round_idx * n_configs)
        for ci in range(round_size):
            cfg = TTC_SAMPLING_CONFIGS[ci]
            try:
                texts = raw_model.generate(batch, tokenizer=tokenizer, **cfg)
            except Exception as e:
                logging.warning(f'[TTC] generate failed for config {cfg}: {e}, skipping')
                continue
            for i, t in enumerate(texts):
                candidates_per_sample[i].append(t)

    mbr_scorer = MBRScorer() if TTC_MBR_WEIGHT > 0 else None
    best_per_sample: List[str] = []
    for i, candidates in enumerate(candidates_per_sample):
        if not candidates:
            best_per_sample.append('')
            continue
        sid = str(batch['ids'][i])
        hclg_entities = hclg_dict.get(sid, [])

        if hclg_entities:
            hclg_scores = verifier.score_batch(candidates, hclg_entities)
        else:
            hclg_scores = [0.0] * len(candidates)

        if mbr_scorer is not None and len(candidates) > 1:
            mbr_scores = mbr_scorer.score_batch(candidates)
        else:
            mbr_scores = [0.0] * len(candidates)

        final_scores = [
            (1.0 - TTC_MBR_WEIGHT) * h + TTC_MBR_WEIGHT * m
            for h, m in zip(hclg_scores, mbr_scores)
        ]
        if max(final_scores) <= 0.0 and not hclg_entities and mbr_scorer is None:
            best_per_sample.append(candidates[0])
            continue
        best_idx = int(max(range(len(final_scores)), key=lambda j: final_scores[j]))
        best_per_sample.append(candidates[best_idx])

    return best_per_sample


@torch.no_grad()
def run_inference(accelerator: Accelerator,
                  ckpt_dir: str,
                  language: str,
                  save_path: str,
                  use_ttc: bool = False,
                  ttc_k: int = TTC_K) -> Tuple[List[str], List[str], List[Any]]:

    tokenizer = AutoTokenizer.from_pretrained(
        BACKBONE_MODEL_NAME,
        trust_remote_code=TRUST_REMOTE_CODE,
        cache_dir=HF_CACHE_DIR,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    test_loader = build_dataloader(
        test_raw_hclg_path, tokenizer,
        mode='test', batch_size=EVAL_BATCH_SIZE,
    )
    if accelerator.is_main_process:
        logging.info('test_loader: %d batches (batch_size=%d)',
                     len(test_loader), EVAL_BATCH_SIZE)

    if accelerator.is_main_process:
        logging.info('Instantiating HCLGAlign... (loading backbone)')
    model = HCLGAlign()
    meta = load_checkpoint(ckpt_dir, model)
    if accelerator.is_main_process:
        logging.info('Loaded checkpoint from %s (meta=%s)', ckpt_dir, meta)

    model, test_loader = accelerator.prepare(model, test_loader)
    raw_model = accelerator.unwrap_model(model)
    model.eval()
    raw_model.eval()

    ttc_verifier: Optional[HCLGVerifier] = None
    test_hclg_dict: Dict[str, List[str]] = {}
    if use_ttc:
        if accelerator.is_main_process:
            logging.info('=' * 70)
            logging.info('TTC ENABLED: K=%d candidates per sample', ttc_k)
            logging.info('  sampling configs (%d): %s', len(TTC_SAMPLING_CONFIGS), TTC_SAMPLING_CONFIGS)
            logging.info('=' * 70)
        test_hclg_dict = load_test_hclg_entities(test_raw_hclg_path)
        if accelerator.is_main_process:
            logging.info('[TTC] loaded test HCLG entities: %d samples', len(test_hclg_dict))

        if TTC_TARGET_LENGTH_AUTO:
            target_length = compute_target_length_from_jsonl(train_raw_hclg_path, max_samples=5000)
        else:
            target_length = TTC_TARGET_LENGTH_FALLBACK
        if accelerator.is_main_process:
            logging.info('[TTC] target_length (impression median tokens) = %.1f', target_length)

        ttc_verifier = HCLGVerifier(
            target_length=target_length,
            length_sigma=TTC_LENGTH_SIGMA,
            length_weight=TTC_LENGTH_WEIGHT,
        )

    local_preds: List[str] = []
    local_refs:  List[str] = []
    local_ids:   List[Any] = []
    desc = 'Test generate (TTC)' if use_ttc else 'Test generate'
    for batch in tqdm(test_loader, desc=desc,
                      disable=not accelerator.is_main_process,
                      dynamic_ncols=True):
        if use_ttc:
            texts = _generate_ttc_batch(
                raw_model, batch, tokenizer,
                verifier=ttc_verifier,
                hclg_dict=test_hclg_dict,
                K=ttc_k,
            )
        else:
            texts = raw_model.generate(batch, tokenizer=tokenizer)
        local_preds.extend(texts)
        local_refs.extend(batch['impression_text'])
        local_ids.extend(batch['ids'])

    all_preds = gather_object(local_preds)
    all_refs  = gather_object(local_refs)
    all_ids   = gather_object(local_ids)

    accelerator.wait_for_everyone()

    if not accelerator.is_main_process:
        return [], [], []

    seen = set()
    preds, refs, ids = [], [], []
    missing_id_count = 0
    for i, r, p in zip(all_ids, all_refs, all_preds):
        if i is None or (isinstance(i, str) and not i.strip()):
            missing_id_count += 1
            key = f'__missing__:{hash((r, p))}'
        else:
            key = str(i)
        if key in seen:
            continue
        seen.add(key)
        ids.append(i); refs.append(r); preds.append(p)
    if missing_id_count > 0:
        logging.warning('Predict dedup: %d samples have missing/empty id, fell back to hash key',
                         missing_id_count)
    logging.info('Generated %d unique predictions (gathered %d, deduped %d)',
                 len(preds), len(all_preds), len(all_preds) - len(preds))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        for sid, ref, pred in zip(ids, refs, preds):
            f.write(json.dumps(
                {'id': sid, 'reference': ref, 'prediction': pred},
                ensure_ascii=False) + '\n')
    logging.info('Saved predictions to %s', save_path)
    return preds, refs, ids


def load_predictions(path: str) -> Tuple[List[str], List[str], List[Any]]:
    """从 test_predictions.jsonl 读回 (preds, refs, ids), 给 --eval-only 用."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'预测文件不存在: {path}\n'
            f'  先生成: python predict.py  (去掉 --eval-only)')
    preds, refs, ids = [], [], []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ids.append(rec['id'])
            refs.append(rec['reference'])
            preds.append(rec['prediction'])
    logging.info('Loaded %d predictions from %s', len(preds), path)
    return preds, refs, ids


_DISTRIBUTED_ENV_KEYS = (
    'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE',
    'GROUP_RANK', 'ROLE_RANK', 'ROLE_WORLD_SIZE',
    'MASTER_ADDR', 'MASTER_PORT',
    'ACCELERATE_PROCESS_INDEX', 'ACCELERATE_LOCAL_PROCESS_INDEX',
    'ACCELERATE_NUM_PROCESSES',
    'TORCHELASTIC_RESTART_COUNT', 'TORCHELASTIC_MAX_RESTARTS',
    'TORCHELASTIC_RUN_ID', 'TORCHELASTIC_USE_AGENT_STORE',
)


def _clean_env_for_eval_subprocess() -> Dict[str, str]:

    env = os.environ.copy()
    for name in _DISTRIBUTED_ENV_KEYS:
        env.pop(name, None)
    env.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
    return env


def _exec_eval_only(args: argparse.Namespace) -> None:

    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        '--eval-only',
        '--language', args.language,
        '--lsc-backend', args.lsc_backend,
    ]
    for attr, flag in (
        ('skip_bertscore', '--skip-bertscore'),
        ('skip_chexbert', '--skip-chexbert'),
        ('skip_radgraph', '--skip-radgraph'),
        ('skip_lsc', '--skip-lsc'),
    ):
        if getattr(args, attr):
            cmd.append(flag)

    logging.info('=' * 70)
    logging.info('Distributed inference finished; exec clean eval-only process')
    logging.info('Command: %s', ' '.join(cmd))
    logging.info('=' * 70)
    logging.shutdown()
    os.execve(sys.executable, cmd, _clean_env_for_eval_subprocess())



def _free_gpu_cache() -> None:

    try:
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


def compute_all_metrics(preds: List[str],
                        refs: List[str],
                        language: str,
                        skip_bertscore: bool = False,
                        skip_chexbert: bool = False,
                        skip_radgraph: bool = False,
                        skip_lsc: bool = False,
                        lsc_backend: str = 'auto') -> Dict[str, Any]:

    assert len(preds) == len(refs), f'preds({len(preds)}) != refs({len(refs)})'
    metrics: Dict[str, Any] = {'n_samples': len(preds), 'language': language}

    logging.info('Computing ROUGE-1/2/L...')
    metrics.update(compute_rouge(preds, refs, language=language))

    if language == 'zh':

        if skip_bertscore:
            logging.info('Skipping BERTScore (--skip-bertscore)')
        else:
            logging.info('Computing BERTScore... (首次会下载 chinese-roberta-wwm-ext)')
            try:
                metrics.update(compute_bertscore(preds, refs, lang=language))
            except Exception as e:
                logging.error('BERTScore failed: %s', e)
                metrics['bertscore_error'] = str(e)
            finally:
                _free_gpu_cache()

        if skip_lsc:
            logging.info('Skipping LSC-F1 (--skip-lsc)')
        else:
            logging.info('Computing LSC-F1 (backend=%s)...', lsc_backend)
            try:
                lsc_cache = os.path.join(result_save_dir, 'lsc_cache.json')
                metrics.update(compute_lsc_f1(
                    preds, refs,
                    backend=lsc_backend,
                    match_mode='jieba_subset',
                    cache_path=lsc_cache,
                    max_model_len=2048,
                ))
            except Exception as e:
                logging.error('LSC-F1 failed: %s', e)
                logging.error('  LSC-F1 (默认 local) 需要: pip install vllm + 能下载/已存 LSC_LLM 模型 '
                              '(config.LSC_LLM_*, 默认 DeepSeek-V2-Lite-Chat, 进程内加载, 建议 3090); '
                              '或 --lsc-backend openai 走 gpt-4o (设 OPENAI_API_KEY); 或 --skip-lsc')
                metrics['lsc_error'] = str(e)

    elif language == 'en':

        if skip_bertscore:
            logging.info('Skipping BERTScore (--skip-bertscore)')
        else:
            logging.info('Computing BERTScore... (首次会下载 roberta-large)')
            try:
                metrics.update(compute_bertscore(preds, refs, lang=language))
            except Exception as e:
                logging.error('BERTScore failed: %s', e)
                metrics['bertscore_error'] = str(e)
            finally:
                _free_gpu_cache()

        if skip_chexbert:
            logging.info('Skipping F1-CheXbert (--skip-chexbert)')
        else:
            logging.info('Computing F1-CheXbert... (首次会下载 chexbert.pth ~500MB)')
            try:
                metrics.update(compute_f1_chexbert(preds, refs))
            except Exception as e:
                logging.error('F1-CheXbert failed: %s', e)
                logging.error('  F1-CheXbert 需要: pip install transformers scikit-learn; '
                              '权重 chexbert.pth 首次自动下到 config.CHEXBERT_CKPT_PATH '
                              '(或手动从 HF StanfordAIMI/RRG_scorers 放进去)')
                metrics['chexbert_error'] = str(e)
            finally:
                _free_gpu_cache()

        if skip_radgraph:
            logging.info('Skipping F1-RadGraph (--skip-radgraph)')
        else:
            logging.info('Computing F1-RadGraph... (首次会下载 radgraph.tar.gz)')
            try:
                metrics.update(compute_f1_radgraph(preds, refs))
            except Exception as e:
                logging.error('F1-RadGraph failed: %s', e)
                logging.error('  F1-RadGraph 需要: pip install radgraph; '
                              '内含老 AllenNLP, 与新 transformers 可能冲突 (见 evaluate.py 注释)')
                metrics['radgraph_error'] = str(e)
            finally:
                _free_gpu_cache()

    else:
        raise ValueError(f"language must be 'zh' or 'en', got {language!r}")

    return metrics


def print_metrics(m: Dict[str, Any]) -> None:

    def pct(key: str) -> str:
        return f'{m[key] * 100:.2f}' if key in m else '  N/A'

    def line(label: str, key: str, marker: str = '') -> str:
        suffix = '   [失败/跳过]' if marker and key not in m else ''
        return f'  {label:<18} {pct(key)}{suffix}'

    lang = m.get('language', '?')
    print('\n' + '=' * 56)
    print(f'  测试集评测结果  (n={m.get("n_samples", "?")}, lang={lang})')
    print('=' * 56)
    print(line('ROUGE-1   (F1)', 'rouge_1_f'))
    print(line('ROUGE-2   (F1)', 'rouge_2_f'))
    print(line('ROUGE-L   (F1)', 'rouge_l_f'))

    if lang == 'zh':
        print(line('BERTScore (F1)', 'bertscore_f', marker='?'))
        print(line('LSC-F1', 'lsc_f1', marker='?'))
    elif lang == 'en':
        print(line('BERTScore (F1)', 'bertscore_f', marker='?'))
        print(line('F1-CheXbert', 'f1_chexbert', marker='?'))
        print(line('F1-RadGraph', 'f1_radgraph', marker='?'))

    print('=' * 56)
    print('  (数值为 ×100; 完整 P/R 等明细见 eval_metrics.json)\n')



def main() -> None:
    parser = argparse.ArgumentParser(
        description='HCLG-Align 测试集推理 + 评测. 指标集按 language 自动选: '
                    'en=ROUGE+BERTScore+F1-CheXbert+F1-RadGraph (MIMIC-CXR); '
                    'zh=ROUGE+BERTScore+LSC-F1 (MC-FUS)')
    parser.add_argument('--ckpt-dir', default=best_ckpt_dir,
                        help='加载哪个 checkpoint 目录 (默认 best_ckpt_dir)')
    parser.add_argument('--language', default=EVAL_LANGUAGE, choices=['zh', 'en'],
                        help='评测语言, 默认走 config.EVAL_LANGUAGE (跟随 config.DATASET 自动切换)')
    parser.add_argument('--eval-only', action='store_true',
                        help='跳过生成, 直接读已存的 test_predictions.jsonl 算指标')
    parser.add_argument('--no-eval', action='store_true',
                        help='只生成预测, 不算指标')
    parser.add_argument('--skip-bertscore', action='store_true',
                        help='跳过 BERTScore (两个数据集都用; 没装 bert-score / 下不了模型时)')
    parser.add_argument('--skip-chexbert', action='store_true',
                        help='跳过 F1-CheXbert (仅 en; 没装 chexbert 权重时)')
    parser.add_argument('--skip-radgraph', action='store_true',
                        help='跳过 F1-RadGraph (仅 en; radgraph 与新 transformers 冲突时)')
    parser.add_argument('--skip-lsc', action='store_true',
                        help='跳过 LSC-F1 (没有 LLM 后端时)')
    parser.add_argument('--lsc-backend', default='local',
                        choices=['auto', 'local', 'openai'],
                        help='LSC-F1 抽取后端: local/auto=进程内 vllm 开源模型 (默认, config.LSC_LLM_*); '
                             'openai=gpt-4o (需 OPENAI_API_KEY)')
    parser.add_argument('--use-ttc', action='store_true', dest='use_ttc',
                        help='启用 Test-Time Compute Scaling: 每个 sample 生成 K 个候选, '
                             '用 HCLG verifier 选最优. 不动训练, 用 config.TTC_K 控制候选数. '
                             '跨数据集 (CXR/FUS) 通用.')
    parser.add_argument('--ttc-k', type=int, default=TTC_K, dest='ttc_k',
                        help=f'TTC 候选数量 (默认 config.TTC_K={TTC_K})')
    args = parser.parse_args()

    run_metrics_in_subprocess = False

    if args.eval_only:
        os.makedirs(result_save_dir, exist_ok=True)
        setup_logging(os.path.join(result_save_dir, 'predict.log'))
        logging.info('=' * 70)
        logging.info('HCLG-Align Eval-only (读已存预测算指标, 不初始化 accelerate)')
        logging.info('=' * 70)
        preds, refs, _ = load_predictions(test_pred_path)
    else:
        accelerator = Accelerator()
        run_metrics_in_subprocess = accelerator.num_processes > 1
        seed_torch(seeds[1])
        if accelerator.is_main_process:
            os.makedirs(result_save_dir, exist_ok=True)
            setup_logging(os.path.join(result_save_dir, 'predict.log'))
            logging.info('=' * 70)
            logging.info('HCLG-Align Test Inference & Evaluation')
            logging.info('=' * 70)
            logging.info('Ckpt dir : %s', args.ckpt_dir)
            logging.info('Test data: %s', test_raw_hclg_path)
            logging.info('Num GPUs : %d', accelerator.num_processes)
        preds, refs, _ = run_inference(
            accelerator, args.ckpt_dir, args.language, test_pred_path,
            use_ttc=args.use_ttc, ttc_k=args.ttc_k)
        if not accelerator.is_main_process:
            return

    if args.no_eval:
        logging.info('Done (--no-eval): predictions saved to %s', test_pred_path)
        return

    if not preds:
        logging.error('没有预测结果, 无法评测.')
        return

    if run_metrics_in_subprocess:
        _exec_eval_only(args)
        return

    metrics = compute_all_metrics(
        preds, refs, args.language,
        skip_bertscore=args.skip_bertscore,
        skip_chexbert=args.skip_chexbert,
        skip_radgraph=args.skip_radgraph,
        skip_lsc=args.skip_lsc,
        lsc_backend=args.lsc_backend,
    )

    with open(eval_metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logging.info('Saved metrics to %s', eval_metrics_path)
    print_metrics(metrics)


if __name__ == '__main__':
    main()
