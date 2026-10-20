from typing import Any, Dict, Optional, Tuple

from config import (
    NUM_MLP_HIDDEN, USE_NODE_TYPE_EMB, NUM_NODE_TYPES,
    RGCN_NUM_LAYERS, RGCN_NUM_BASES, RGCN_DROPOUT,
    PROJ_HIDDEN, PROJ_DROPOUT,
    NUM_RELATIONS, D_G,
    BACKBONE_MODEL_NAME, BACKBONE_DTYPE, TRUST_REMOTE_CODE, HF_CACHE_DIR,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    INFONCE_TAU_INIT, INFONCE_TAU_LEARNABLE,
    IGNORE_INDEX,
    USE_GRADIENT_CHECKPOINTING,
    GRAPH_DROPOUT_P,
    USE_PREFIX_TUNING, PREFIX_PROJ_DROPOUT,
    GRAPH_FUSION_MODE, CROSS_ATTN_EVERY, CROSS_ATTN_NUM_HEADS, CROSS_ATTN_DROPOUT,
    GEN_MAX_NEW_TOKENS, GEN_NUM_BEAMS, GEN_DO_SAMPLE, GEN_REPETITION_PENALTY,
    GEN_LENGTH_PENALTY,
)

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.utils import scatter, to_dense_batch, softmax as pyg_softmax

try:
    from liger_kernel.transformers import apply_liger_kernel_to_qwen3
    apply_liger_kernel_to_qwen3(
        rope=True,
        rms_norm=True,
        swiglu=True,
        fused_linear_cross_entropy=True,
        cross_entropy=False,
    )
    print('[model.py] Liger Kernel applied to Qwen3 (FLCE on, expect -1.5GB VRAM)',
          flush=True)
except ImportError:
    print('[model.py] WARNING: liger-kernel not installed. '
          'Logits memory will not be optimized. '
          'Run `pip install liger-kernel` to fix VRAM pressure.',
          flush=True)

from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType



class NumMLP(nn.Module):

    def __init__(self, out_dim: int, hidden_dim: int = NUM_MLP_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [k, 1]  ->  [k, out_dim]"""
        return self.net(x)



class NodeInitializer(nn.Module):

    def __init__(self,
                 d_init: int,
                 use_type_emb: bool = USE_NODE_TYPE_EMB,
                 num_node_types: int = NUM_NODE_TYPES,
                 num_mlp_hidden: int = NUM_MLP_HIDDEN):

        super().__init__()
        self.d_init = d_init
        self.num_mlp = NumMLP(out_dim=d_init, hidden_dim=num_mlp_hidden)
        self.use_type_emb = use_type_emb
        if use_type_emb:
            self.type_emb = nn.Embedding(num_node_types, d_init)

    def forward(self, graph_batch, embed_tokens) -> torch.Tensor:

        ids  = graph_batch.node_text_ids
        mask = graph_batch.node_text_mask

        emb = embed_tokens(ids).float()

        m = mask.unsqueeze(-1).float()
        x_text = (emb * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)

        flat_v = graph_batch.flat_values
        seg_id = graph_batch.value_seg_ids

        if flat_v.numel() > 0:
            num_emb = self.num_mlp(flat_v.unsqueeze(-1))
            num_per_node = scatter(num_emb, seg_id, dim=0,
                                    dim_size=x_text.shape[0], reduce='mean')
            x = x_text + num_per_node
        else:
            x = x_text

        if self.use_type_emb:
            x = x + self.type_emb(graph_batch.node_type)

        return x



class RGCNEncoder(nn.Module):

    def __init__(self,
                 d_in: int,
                 d_hidden: int = D_G,
                 num_relations: int = NUM_RELATIONS,
                 num_layers: int = RGCN_NUM_LAYERS,
                 num_bases = RGCN_NUM_BASES,
                 dropout: float = RGCN_DROPOUT):

        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(RGCNConv(d_in, d_hidden, num_relations,
                                    num_bases=num_bases))
        for _ in range(num_layers - 1):
            self.convs.append(RGCNConv(d_hidden, d_hidden, num_relations,
                                        num_bases=num_bases))
        self.norms = nn.ModuleList([nn.LayerNorm(d_hidden)
                                     for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_type) -> torch.Tensor:

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_type)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)
        return x



class ProjectionHead(nn.Module):

    def __init__(self,
                 d_in: int,
                 d_out: int,
                 d_hidden = PROJ_HIDDEN,
                 dropout: float = PROJ_DROPOUT):
        super().__init__()
        if d_hidden is None:
            d_hidden = d_out
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
            nn.LayerNorm(d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)





class LayerGraphInjector(nn.Module):

    def __init__(self,
                 d_llm: int,
                 num_layers: int,
                 dropout: float = 0.0):
        super().__init__()
        self.num_layers = num_layers
        self.d_llm = d_llm
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_llm),
                nn.Linear(d_llm, d_llm),
                nn.GELU(),
                nn.Linear(d_llm, d_llm),
            )
            for _ in range(num_layers)
        ])
        self.layer_scales = nn.Parameter(torch.zeros(num_layers))

    def get_bias(self, z_pos: torch.Tensor, layer_idx: int) -> torch.Tensor:

        bias = self.projections[layer_idx](z_pos)
        scale = torch.tanh(self.layer_scales[layer_idx])
        return (bias * scale).unsqueeze(1)


class GraphCrossAttnInjector(nn.Module):

    def __init__(self,
                 d_llm: int,
                 num_hidden_layers: int,
                 every: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.0):
        super().__init__()
        self.layer_ids = [i for i in range(num_hidden_layers)
                          if i % every == (every - 1)]
        if not self.layer_ids:
            self.layer_ids = [num_hidden_layers - 1]
        self.attn   = nn.ModuleDict()
        self.q_norm = nn.ModuleDict()
        self.gates  = nn.ParameterDict()
        for i in self.layer_ids:
            self.attn[str(i)]   = nn.MultiheadAttention(
                d_llm, num_heads, dropout=dropout, batch_first=True)
            self.q_norm[str(i)] = nn.LayerNorm(d_llm)
            self.gates[str(i)]  = nn.Parameter(torch.zeros(1))
        self.last_id = self.layer_ids[-1]

    def has_layer(self, idx: int) -> bool:
        return str(idx) in self.attn

    def get_delta(self,
                  hidden: torch.Tensor,
                  kv: torch.Tensor,
                  key_padding_mask: torch.Tensor,
                  idx: int) -> torch.Tensor:
        q = self.q_norm[str(idx)](hidden)
        out, _ = self.attn[str(idx)](
            q, kv, kv,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return out * torch.tanh(self.gates[str(idx)])


class AttentionPool(nn.Module):

    def __init__(self, d_in: int):
        super().__init__()
        self.attn = nn.Linear(d_in, 1, bias=False)
        nn.init.zeros_(self.attn.weight)

    def forward(self,
                 E: torch.Tensor,
                 batch_idx: torch.Tensor,
                 ) -> torch.Tensor:
        scores = self.attn(E.float()).squeeze(-1)
        weights = pyg_softmax(scores, batch_idx)
        weighted = E * weights.unsqueeze(-1).to(E.dtype)
        return scatter(weighted, batch_idx, dim=0, reduce='sum')


class HCLGAlign(nn.Module):

    def __init__(self,
                 backbone_name: str = BACKBONE_MODEL_NAME,
                 backbone_dtype: str = BACKBONE_DTYPE,
                 trust_remote_code: bool = TRUST_REMOTE_CODE,
                 cache_dir: Optional[str] = HF_CACHE_DIR,
                 lora_r: int = LORA_R,
                 lora_alpha: int = LORA_ALPHA,
                 lora_dropout: float = LORA_DROPOUT,
                 lora_targets = LORA_TARGET_MODULES,
                 d_g: int = D_G,
                 num_relations: int = NUM_RELATIONS,
                 num_layers: int = RGCN_NUM_LAYERS,
                 num_bases = RGCN_NUM_BASES,
                 rgcn_dropout: float = RGCN_DROPOUT,
                 use_type_emb: bool = USE_NODE_TYPE_EMB,
                 num_node_types: int = NUM_NODE_TYPES,
                 num_mlp_hidden: int = NUM_MLP_HIDDEN,
                 proj_hidden = PROJ_HIDDEN,
                 proj_dropout: float = PROJ_DROPOUT,
                 tau_init: float = INFONCE_TAU_INIT,
                 tau_learnable: bool = INFONCE_TAU_LEARNABLE):
        super().__init__()

        _local_rank_str = os.environ.get('LOCAL_RANK')
        _device_map = {"": int(_local_rank_str)} if _local_rank_str is not None else None
        self.backbone = AutoModelForCausalLM.from_pretrained(
            backbone_name,
            torch_dtype=getattr(torch, backbone_dtype),
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
            device_map=_device_map,
        )

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=list(lora_targets),
            bias='none',
            task_type=TaskType.CAUSAL_LM,
        )
        self.backbone = get_peft_model(self.backbone, lora_config)

        if USE_GRADIENT_CHECKPOINTING:
            self.backbone.config.use_cache = False

            self.backbone.gradient_checkpointing_enable()

            self.backbone.enable_input_require_grads()

        self.d_init = self.backbone.config.hidden_size
        self.d_llm = self.d_init
        self.d_g = d_g

        self.node_init = NodeInitializer(
            d_init=self.d_init,
            use_type_emb=use_type_emb,
            num_node_types=num_node_types,
            num_mlp_hidden=num_mlp_hidden,
        )
        self.rgcn = RGCNEncoder(
            d_in=self.d_init,
            d_hidden=d_g,
            num_relations=num_relations,
            num_layers=num_layers,
            num_bases=num_bases,
            dropout=rgcn_dropout,
        )
        self.proj = ProjectionHead(
            d_in=d_g,
            d_out=self.d_llm,
            d_hidden=proj_hidden,
            dropout=proj_dropout,
        )
        self.readout = AttentionPool(d_in=self.d_llm)

        log_scale_init = torch.log(torch.tensor(1.0 / tau_init, dtype=torch.float32))
        if tau_learnable:
            self.log_logit_scale = nn.Parameter(log_scale_init)
        else:
            self.register_buffer('log_logit_scale', log_scale_init)

        self._inject_z_pos = None
        self._inject_z_pos_with_grad = None
        self._inject_node_kv = None
        self._inject_node_kv_grad = None
        self._inject_node_mask = None
        self._inject_active = False

        self._use_injector  = USE_PREFIX_TUNING and GRAPH_FUSION_MODE in ('inject', 'both')
        self._use_crossattn = USE_PREFIX_TUNING and GRAPH_FUSION_MODE in ('crossattn', 'both')
        self._use_hook_path = self._use_injector or self._use_crossattn

        _dtype = getattr(torch, backbone_dtype)

        if self._use_injector:
            num_hidden_layers = self.backbone.config.num_hidden_layers
            self.layer_injector = LayerGraphInjector(
                d_llm=self.d_llm, num_layers=num_hidden_layers,
                dropout=PREFIX_PROJ_DROPOUT,
            ).to(_dtype)
            print(f'[model.py] LayerGraphInjector enabled (mode={GRAPH_FUSION_MODE}): '
                  f'n_layers={num_hidden_layers}, d_llm={self.d_llm}', flush=True)
        else:
            self.layer_injector = None

        if self._use_crossattn:
            num_hidden_layers = self.backbone.config.num_hidden_layers
            self.cross_attn = GraphCrossAttnInjector(
                d_llm=self.d_llm, num_hidden_layers=num_hidden_layers,
                every=CROSS_ATTN_EVERY, num_heads=CROSS_ATTN_NUM_HEADS,
                dropout=CROSS_ATTN_DROPOUT,
            ).to(_dtype)
            print(f'[model.py] GraphCrossAttnInjector enabled (mode={GRAPH_FUSION_MODE}): '
                  f'CA layers={self.cross_attn.layer_ids}, heads={CROSS_ATTN_NUM_HEADS}, '
                  f'last_grad_layer={self.cross_attn.last_id}', flush=True)
        else:
            self.cross_attn = None

        if self._use_hook_path:
            num_hidden_layers = self.backbone.config.num_hidden_layers
            _layers = None
            for _path in [
                lambda m: m.base_model.model.model.layers,
                lambda m: m.base_model.model.layers,
                lambda m: m.model.model.layers,
                lambda m: m.model.layers,
            ]:
                try:
                    _candidate = _path(self.backbone)
                    if isinstance(_candidate, nn.ModuleList) and len(_candidate) == num_hidden_layers:
                        _layers = _candidate
                        break
                except AttributeError:
                    continue
            if _layers is None:
                for _name, _module in self.backbone.named_modules():
                    if isinstance(_module, nn.ModuleList) and len(_module) == num_hidden_layers:
                        if hasattr(_module[0], 'self_attn'):
                            _layers = _module
                            print(f'[model.py] 兜底找到 layers at: {_name}', flush=True)
                            break

            if _layers is None:
                raise RuntimeError(
                    "无法找到 transformer layers 来注册 hook. "
                    "请检查 PEFT 包装结构 (self.backbone.named_modules())."
                )

            self._injector_hooks = []
            _last_idx = num_hidden_layers - 1
            for _layer_idx, _layer in enumerate(_layers):
                def _make_hook(idx):
                    def _hook(_module, _inputs, _output):
                        if not self._inject_active:
                            return _output
                        _h = _output[0] if isinstance(_output, tuple) else _output

                        if self.layer_injector is not None:
                            if idx == _last_idx and self._inject_z_pos_with_grad is not None:
                                _z = self._inject_z_pos_with_grad
                            else:
                                _z = self._inject_z_pos
                            if _z is not None:
                                if _z.shape[0] != _h.shape[0]:
                                    _f = _h.shape[0] // _z.shape[0]
                                    _z = _z.repeat_interleave(_f, dim=0) \
                                        if _f * _z.shape[0] == _h.shape[0] else None
                                if _z is not None:
                                    _h = _h + self.layer_injector.get_bias(_z, idx).to(_h.dtype)

                        if self.cross_attn is not None and self.cross_attn.has_layer(idx):
                            if idx == self.cross_attn.last_id and self._inject_node_kv_grad is not None:
                                _kv = self._inject_node_kv_grad
                            else:
                                _kv = self._inject_node_kv
                            _mask = self._inject_node_mask
                            if _kv is not None and _mask is not None:
                                if _kv.shape[0] != _h.shape[0]:
                                    _f = _h.shape[0] // _kv.shape[0]
                                    if _f * _kv.shape[0] == _h.shape[0]:
                                        _kv = _kv.repeat_interleave(_f, dim=0)
                                        _mask = _mask.repeat_interleave(_f, dim=0)
                                    else:
                                        _kv = None
                                if _kv is not None:
                                    _delta = self.cross_attn.get_delta(
                                        _h, _kv.to(_h.dtype), ~_mask, idx)
                                    _h = _h + _delta.to(_h.dtype)

                        if isinstance(_output, tuple):
                            return (_h,) + _output[1:]
                        return _h
                    return _hook
                _handle = _layer.register_forward_hook(_make_hook(_layer_idx))
                self._injector_hooks.append(_handle)
            print(f'[model.py] 已注册 {len(self._injector_hooks)} 个 forward hook '
                  f'(injector={self.layer_injector is not None}, '
                  f'crossattn={self.cross_attn is not None})', flush=True)
        else:
            self.layer_injector = None
            self.cross_attn = None
            print('[model.py] 图 hook 路径 DISABLED (USE_PREFIX_TUNING=False), '
                  'falling back to v5 input-concat graph-token path', flush=True)


    def _encode_graph(self, graph_batch) -> torch.Tensor:

        embed_tokens = self.backbone.get_input_embeddings()
        x_init = self.node_init(graph_batch, embed_tokens)
        h = self.rgcn(x_init,
                       graph_batch.edge_index,
                       graph_batch.edge_type)
        E_graph = self.proj(h)
        return E_graph

    def _compute_z_text(self,
                         findings_ids: torch.Tensor,
                         findings_mask: torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            outputs = self.backbone(
                input_ids=findings_ids,
                attention_mask=findings_mask.long(),
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = outputs.hidden_states[-1]

            last_pos = findings_mask.long().sum(dim=1) - 1
            batch_idx = torch.arange(findings_ids.shape[0], device=findings_ids.device)
            z_text = hidden[batch_idx, last_pos, :]
        return z_text

    def _readout(self,
                  E_graph: torch.Tensor,
                  graph_batch_idx: torch.Tensor) -> torch.Tensor:

        return self.readout(E_graph, graph_batch_idx)

    def _info_nce(self,
                   z_text: torch.Tensor,
                   z_pos: torch.Tensor,
                   z_neg: torch.Tensor) -> torch.Tensor:

        z_text = z_text.float()
        z_pos = z_pos.float()
        z_neg = z_neg.float()

        z_text = F.normalize(z_text, dim=-1)
        z_pos = F.normalize(z_pos, dim=-1)
        z_neg = F.normalize(z_neg, dim=-1)

        logit_scale = self.log_logit_scale.exp().clamp(max=100.0)

        sim_pos = (z_text * z_pos).sum(dim=-1) * logit_scale
        sim_neg = torch.einsum('bd,bkd->bk', z_text, z_neg) * logit_scale

        all_logits = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1)
        loss = -sim_pos + torch.logsumexp(all_logits, dim=1)
        return loss.mean()

    def _build_train_inputs(self,
                              E_pos_dense: torch.Tensor,
                              graph_mask: torch.Tensor,
                              findings_ids: torch.Tensor,
                              findings_mask: torch.Tensor,
                              impression_ids: torch.Tensor,
                              impression_mask: torch.Tensor,
                              prompt_pre_ids: torch.Tensor,
                              prompt_suf_ids: torch.Tensor,
                              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        device = E_pos_dense.device
        embed_tokens = self.backbone.get_input_embeddings()

        pre_embed = embed_tokens(prompt_pre_ids)
        suf_embed = embed_tokens(prompt_suf_ids)
        findings_embed = embed_tokens(findings_ids)
        impression_embed = embed_tokens(impression_ids)

        text_dtype = pre_embed.dtype
        E_pos_dense = E_pos_dense.to(text_dtype)

        N_per = graph_mask.sum(dim=1)
        M_per = findings_mask.long().sum(dim=1)
        L_per = impression_mask.long().sum(dim=1)
        P_pre = prompt_pre_ids.shape[0]
        P_suf = prompt_suf_ids.shape[0]

        total_per = N_per + P_pre + M_per + P_suf + L_per
        T_max = int(total_per.max().item())

        B = findings_ids.shape[0]
        d_llm = E_pos_dense.shape[-1]

        inputs_embeds = torch.zeros(B, T_max, d_llm, dtype=text_dtype, device=device)
        attention_mask = torch.zeros(B, T_max, dtype=torch.long, device=device)
        labels = torch.full((B, T_max), IGNORE_INDEX, dtype=torch.long, device=device)

        for b in range(B):
            n = int(N_per[b].item())
            m = int(M_per[b].item())
            l = int(L_per[b].item())

            off = 0
            inputs_embeds[b, off:off+n] = E_pos_dense[b, :n]
            off += n
            inputs_embeds[b, off:off+P_pre] = pre_embed
            off += P_pre
            inputs_embeds[b, off:off+m] = findings_embed[b, :m]
            off += m
            inputs_embeds[b, off:off+P_suf] = suf_embed
            off += P_suf
            inputs_embeds[b, off:off+l] = impression_embed[b, :l]
            labels[b, off:off+l] = impression_ids[b, :l]
            off += l

            attention_mask[b, :off] = 1

        return inputs_embeds, attention_mask, labels

    def _build_inference_inputs(self,
                                  E_pos_dense: torch.Tensor,
                                  graph_mask: torch.Tensor,
                                  findings_ids: torch.Tensor,
                                  findings_mask: torch.Tensor,
                                  prompt_pre_ids: torch.Tensor,
                                  prompt_suf_ids: torch.Tensor,
                                  ) -> Tuple[torch.Tensor, torch.Tensor]:

        device = E_pos_dense.device
        embed_tokens = self.backbone.get_input_embeddings()

        pre_embed = embed_tokens(prompt_pre_ids)
        suf_embed = embed_tokens(prompt_suf_ids)
        findings_embed = embed_tokens(findings_ids)

        text_dtype = pre_embed.dtype
        E_pos_dense = E_pos_dense.to(text_dtype)

        N_per = graph_mask.sum(dim=1)
        M_per = findings_mask.long().sum(dim=1)
        P_pre = prompt_pre_ids.shape[0]
        P_suf = prompt_suf_ids.shape[0]

        total_per = N_per + P_pre + M_per + P_suf
        T_max = int(total_per.max().item())

        B = findings_ids.shape[0]
        d_llm = E_pos_dense.shape[-1]

        inputs_embeds = torch.zeros(B, T_max, d_llm, dtype=text_dtype, device=device)
        attention_mask = torch.zeros(B, T_max, dtype=torch.long, device=device)

        for b in range(B):
            n = int(N_per[b].item())
            m = int(M_per[b].item())
            total = n + P_pre + m + P_suf
            pad_len = T_max - total

            off = pad_len
            inputs_embeds[b, off:off+n] = E_pos_dense[b, :n]
            off += n
            inputs_embeds[b, off:off+P_pre] = pre_embed
            off += P_pre
            inputs_embeds[b, off:off+m] = findings_embed[b, :m]
            off += m
            inputs_embeds[b, off:off+P_suf] = suf_embed

            attention_mask[b, pad_len:] = 1

        return inputs_embeds, attention_mask


    def _build_text_only_train_inputs(self,
                                       findings_ids: torch.Tensor,
                                       findings_mask: torch.Tensor,
                                       impression_ids: torch.Tensor,
                                       impression_mask: torch.Tensor,
                                       prompt_pre_ids: torch.Tensor,
                                       prompt_suf_ids: torch.Tensor,
                                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        device = findings_ids.device

        M_per = findings_mask.long().sum(dim=1)
        L_per = impression_mask.long().sum(dim=1)
        P_pre = prompt_pre_ids.shape[0]
        P_suf = prompt_suf_ids.shape[0]

        total_per = P_pre + M_per + P_suf + L_per
        T_max = int(total_per.max().item())

        B = findings_ids.shape[0]

        input_ids = torch.zeros(B, T_max, dtype=torch.long, device=device)
        attention_mask = torch.zeros(B, T_max, dtype=torch.long, device=device)
        labels = torch.full((B, T_max), IGNORE_INDEX, dtype=torch.long, device=device)

        for b in range(B):
            m = int(M_per[b].item())
            l = int(L_per[b].item())

            off = 0
            input_ids[b, off:off+P_pre] = prompt_pre_ids
            off += P_pre
            input_ids[b, off:off+m] = findings_ids[b, :m]
            off += m
            input_ids[b, off:off+P_suf] = prompt_suf_ids
            off += P_suf
            input_ids[b, off:off+l] = impression_ids[b, :l]
            labels[b, off:off+l] = impression_ids[b, :l]
            off += l

            attention_mask[b, :off] = 1

        return input_ids, attention_mask, labels

    def _build_text_only_inference_inputs(self,
                                            findings_ids: torch.Tensor,
                                            findings_mask: torch.Tensor,
                                            prompt_pre_ids: torch.Tensor,
                                            prompt_suf_ids: torch.Tensor,
                                            pad_id: int,
                                            ) -> Tuple[torch.Tensor, torch.Tensor]:

        device = findings_ids.device

        M_per = findings_mask.long().sum(dim=1)
        P_pre = prompt_pre_ids.shape[0]
        P_suf = prompt_suf_ids.shape[0]

        total_per = P_pre + M_per + P_suf
        T_max = int(total_per.max().item())

        B = findings_ids.shape[0]

        input_ids = torch.full((B, T_max), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros(B, T_max, dtype=torch.long, device=device)

        for b in range(B):
            m = int(M_per[b].item())
            total = P_pre + m + P_suf
            pad_len = T_max - total

            off = pad_len
            input_ids[b, off:off+P_pre] = prompt_pre_ids
            off += P_pre
            input_ids[b, off:off+m] = findings_ids[b, :m]
            off += m
            input_ids[b, off:off+P_suf] = prompt_suf_ids

            attention_mask[b, pad_len:] = 1

        return input_ids, attention_mask


    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:

        pos_batch = batch['pos_graph']
        neg_batch = batch.get('neg_graph')
        findings_ids = batch['findings_ids']
        findings_mask = batch['findings_mask']
        impression_ids = batch.get('impression_ids')
        impression_mask = batch.get('impression_mask')
        prompt_pre_ids = batch['prompt_pre_ids']
        prompt_suf_ids = batch['prompt_suf_ids']
        K = batch.get('k_neg', 0)

        device = findings_ids.device
        B = findings_ids.shape[0]

        self._inject_z_pos = None
        self._inject_z_pos_with_grad = None
        self._inject_node_kv = None
        self._inject_node_kv_grad = None
        self._inject_node_mask = None
        self._inject_active = False

        E_pos = self._encode_graph(pos_batch)

        z_text = self._compute_z_text(findings_ids, findings_mask)

        z_pos = self._readout(E_pos, pos_batch.batch)

        if neg_batch is not None and K > 0:
            E_neg = self._encode_graph(neg_batch)
            z_neg = self._readout(E_neg, neg_batch.batch)
            z_neg = z_neg.view(B, K, -1)
            L_cl = self._info_nce(z_text, z_pos, z_neg)
        else:
            L_cl = torch.tensor(0.0, device=device)

        if impression_ids is not None:
            if self._use_hook_path:

                backbone_dtype = next(self.backbone.parameters()).dtype
                if self._use_injector:
                    E_pos_for_hook = self._encode_graph(pos_batch)
                    z_pos_for_hook = self._readout(E_pos_for_hook, pos_batch.batch).to(backbone_dtype)
                    self._inject_z_pos = z_pos_for_hook.detach()
                    self._inject_z_pos_with_grad = z_pos_for_hook
                if self._use_crossattn:
                    E_pos_for_ca = self._encode_graph(pos_batch)
                    E_ca_dense, node_mask = to_dense_batch(E_pos_for_ca, pos_batch.batch)
                    E_ca_dense = E_ca_dense.to(backbone_dtype)
                    self._inject_node_kv = E_ca_dense.detach()
                    self._inject_node_kv_grad = E_ca_dense
                    self._inject_node_mask = node_mask

                input_ids, attention_mask, labels = self._build_text_only_train_inputs(
                    findings_ids, findings_mask,
                    impression_ids, impression_mask,
                    prompt_pre_ids, prompt_suf_ids,
                )

                self._inject_active = True
                outputs = self.backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_dict=True,
                )
                L_gen = outputs.loss
            else:
                E_pos_dense, graph_mask = to_dense_batch(E_pos, pos_batch.batch)

                if self.training and GRAPH_DROPOUT_P > 0:
                    keep_per_sample = torch.rand(B, device=device) > GRAPH_DROPOUT_P
                    graph_mask = graph_mask & keep_per_sample.view(B, 1)

                inputs_embeds, attention_mask, labels = self._build_train_inputs(
                    E_pos_dense, graph_mask,
                    findings_ids, findings_mask,
                    impression_ids, impression_mask,
                    prompt_pre_ids, prompt_suf_ids,
                )

                outputs = self.backbone(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_dict=True,
                )
                L_gen = outputs.loss
        else:
            L_gen = torch.tensor(0.0, device=device)

        return {
            'L_gen': L_gen,
            'L_cl': L_cl,
            'tau': self.log_logit_scale.exp().reciprocal().detach(),
            'n_pos_nodes': pos_batch.num_nodes,
            'n_neg_nodes': neg_batch.num_nodes if neg_batch is not None else 0,
        }


    @torch.no_grad()
    def generate(self,
                  batch: Dict[str, Any],
                  tokenizer = None,
                  **gen_kwargs):

        self.eval()

        pos_batch = batch['pos_graph']
        findings_ids = batch['findings_ids']
        findings_mask = batch['findings_mask']
        prompt_pre_ids = batch['prompt_pre_ids']
        prompt_suf_ids = batch['prompt_suf_ids']
        pad_id = batch['pad_id']

        E_pos = self._encode_graph(pos_batch)

        if tokenizer is not None and tokenizer.eos_token_id is not None:
            eos_id = tokenizer.eos_token_id
        else:
            eos_id = self.backbone.config.eos_token_id
        gen_args = dict(
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            num_beams=GEN_NUM_BEAMS,
            do_sample=GEN_DO_SAMPLE,
            repetition_penalty=GEN_REPETITION_PENALTY,
            length_penalty=GEN_LENGTH_PENALTY,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
            return_dict_in_generate=False,
        )
        gen_args.update(gen_kwargs)

        if self._use_hook_path:
            try:
                backbone_dtype = next(self.backbone.parameters()).dtype
                if self._use_injector:
                    z_pos = self._readout(E_pos, pos_batch.batch)
                    self._inject_z_pos = z_pos.to(backbone_dtype)
                    self._inject_z_pos_with_grad = None
                if self._use_crossattn:
                    E_ca_dense, node_mask = to_dense_batch(E_pos, pos_batch.batch)
                    self._inject_node_kv = E_ca_dense.to(backbone_dtype)
                    self._inject_node_kv_grad = None
                    self._inject_node_mask = node_mask
                self._inject_active = True

                input_ids, attention_mask = self._build_text_only_inference_inputs(
                    findings_ids, findings_mask,
                    prompt_pre_ids, prompt_suf_ids,
                    pad_id=pad_id,
                )

                output_ids = self.backbone.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_args,
                )
                prefix_len = input_ids.shape[1]
                output_ids = output_ids[:, prefix_len:]
            finally:
                self._inject_z_pos = None
                self._inject_z_pos_with_grad = None
                self._inject_node_kv = None
                self._inject_node_kv_grad = None
                self._inject_node_mask = None
                self._inject_active = False
        else:
            E_pos_dense, graph_mask = to_dense_batch(E_pos, pos_batch.batch)
            inputs_embeds, attention_mask = self._build_inference_inputs(
                E_pos_dense, graph_mask,
                findings_ids, findings_mask,
                prompt_pre_ids, prompt_suf_ids,
            )
            output_ids = self.backbone.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **gen_args,
            )

        if tokenizer is not None:
            return tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return output_ids



if __name__ == '__main__':
    from transformers import AutoTokenizer
    from data import build_dataloader
    from config import (BACKBONE_MODEL_NAME, TRUST_REMOTE_CODE, HF_CACHE_DIR,
                         test_raw_hclg_path)

    print('=' * 70)
    print('Phase 1 self-test: 纯 GNN 侧 (不加载 8B backbone)')
    print('=' * 70)

    d_init_test = 512
    d_g_test    = 256
    d_llm_test  = 512

    tok = AutoTokenizer.from_pretrained(
        BACKBONE_MODEL_NAME, trust_remote_code=TRUST_REMOTE_CODE,
        cache_dir=HF_CACHE_DIR,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dummy_embed = nn.Embedding(tok.vocab_size, d_init_test)

    loader = build_dataloader(test_raw_hclg_path, tok, mode='test',
                               batch_size=2, num_workers=0)
    batch = next(iter(loader))
    pos_batch = batch['pos_graph']
    print(f'\n[Input pos_batch]')
    print(f'  num_graphs       = {pos_batch.num_graphs}')
    print(f'  total_N          = {pos_batch.num_nodes}')
    print(f'  edge_index.shape = {tuple(pos_batch.edge_index.shape)}')
    print(f'  edge_type.shape  = {tuple(pos_batch.edge_type.shape)}')
    print(f'  node_text_ids    = {tuple(pos_batch.node_text_ids.shape)}')
    print(f'  flat_values      = {tuple(pos_batch.flat_values.shape)}')
    print(f'  value_seg_ids    = {tuple(pos_batch.value_seg_ids.shape)}')

    node_init = NodeInitializer(d_init=d_init_test)
    rgcn      = RGCNEncoder(d_in=d_init_test, d_hidden=d_g_test)
    proj      = ProjectionHead(d_in=d_g_test, d_out=d_llm_test)

    print(f'\n[Stage 1] NodeInitializer (text embed + NumMLP scatter pool)')
    x_init = node_init(pos_batch, dummy_embed)
    print(f'  x_init.shape = {tuple(x_init.shape)}'
           f'   expected = ({pos_batch.num_nodes}, {d_init_test})')
    assert x_init.shape == (pos_batch.num_nodes, d_init_test)

    print(f'\n[Stage 2] RGCNEncoder (R-GCN x {RGCN_NUM_LAYERS} layers, '
           f'{NUM_RELATIONS} relations)')
    H = rgcn(x_init, pos_batch.edge_index, pos_batch.edge_type)
    print(f'  H.shape = {tuple(H.shape)}'
           f'   expected = ({pos_batch.num_nodes}, {d_g_test})')
    assert H.shape == (pos_batch.num_nodes, d_g_test)

    print(f'\n[Stage 3] ProjectionHead (d_g -> d_llm)')
    E_graph = proj(H)
    print(f'  E_graph.shape = {tuple(E_graph.shape)}'
           f'   expected = ({pos_batch.num_nodes}, {d_llm_test})')
    assert E_graph.shape == (pos_batch.num_nodes, d_llm_test)

    print(f'\n[Parameter count]')
    n_num = sum(p.numel() for p in node_init.num_mlp.parameters())
    n_ni  = sum(p.numel() for p in node_init.parameters())
    n_rg  = sum(p.numel() for p in rgcn.parameters())
    n_pj  = sum(p.numel() for p in proj.parameters())
    print(f'  NumMLP                : {n_num:>10,d}')
    print(f'  NodeInitializer (含 NumMLP): {n_ni:>10,d}')
    print(f'  RGCNEncoder           : {n_rg:>10,d}')
    print(f'  ProjectionHead        : {n_pj:>10,d}')
    print(f'  TOTAL (Phase 1)       : {n_ni + n_rg + n_pj:>10,d}')

    print(f'\n所有 shape 一致, 阶段 1 通过 ✓')
    print(f'下一步: 阶段 2 接入 Qwen3-8B + LoRA + 损失计算')
