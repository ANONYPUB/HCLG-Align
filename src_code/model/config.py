import logging
import os
import sys
import math

_HF_MIRROR = 'https://hf-mirror.com'
os.environ['HF_ENDPOINT']     = _HF_MIRROR
os.environ['HF_HUB_ENDPOINT'] = _HF_MIRROR
os.environ['HF_HUB_DISABLE_XET'] = '1'


import warnings as _warnings
_warnings.filterwarnings('ignore', message='.*An issue occurred while importing.*')
_warnings.filterwarnings('ignore', message='.*pkg_resources is deprecated.*')


def _looks_like_plain_hf_model_dir(path: str) -> bool:
    """识别 ModelScope/HF --local_dir 下载的普通本地模型目录."""
    if not os.path.isdir(path):
        return False
    has_config = os.path.isfile(os.path.join(path, 'config.json'))
    has_weights = any(
        os.path.exists(os.path.join(path, name))
        for name in (
            'model.safetensors',
            'model.safetensors.index.json',
            'pytorch_model.bin',
            'pytorch_model.bin.index.json',
        )
    )
    return has_config and has_weights


DATASET = 'CXR'

if DATASET == 'FUS':
    top_data_dir        = '../../data/FUS'
    BUILD_HCLG_DATASET  = 'mc_fus'
    EVAL_LANGUAGE       = 'zh'
    BACKBONE_MODEL_NAME = 'Qwen/Qwen3-0.6B'
    PROMPT_PREFIX       = '检查所见：'
    PROMPT_SUFFIX       = '\n诊断结论：'
    MODEL_VERSION       = 'v9.5-mini'
    TRAIN_BATCH_SIZE    = 64
    EVAL_BATCH_SIZE     = 16
    K_NEG_PER_STEP      = 3
    EDGE_DROPOUT_P      = 0.2
    EVAL_ROUGE_START_EPOCH = 5
    NUM_EPOCHS      = 15
    GEN_MAX_NEW_TOKENS   = 128
    GEN_NUM_BEAMS        = 4
    GEN_REPETITION_PENALTY = 1.05
    GEN_LENGTH_PENALTY     = 1.1
    MAX_FINDINGS_TOKENS   = 768
    MAX_IMPRESSION_TOKENS = 128
    MAX_NODE_TEXT_TOKENS  = 32
    GRAD_ACCUM_STEPS = 1
elif DATASET == 'CXR':
    top_data_dir        = '../../data/CXR'
    BUILD_HCLG_DATASET  = 'mimic_cxr'
    EVAL_LANGUAGE       = 'en'
    BACKBONE_MODEL_NAME = 'Qwen/Qwen3-0.6B'
    PROMPT_PREFIX       = 'Findings: '
    PROMPT_SUFFIX       = '\nImpression: '
    MODEL_VERSION       = 'v9.4-mini'
    TRAIN_BATCH_SIZE    = 224
    EVAL_BATCH_SIZE     = 32
    K_NEG_PER_STEP      = 3
    EDGE_DROPOUT_P      = 0.2
    EVAL_ROUGE_START_EPOCH = 1
    NUM_EPOCHS      = 15
    GEN_MAX_NEW_TOKENS   = 128
    GEN_NUM_BEAMS        = 4
    GEN_REPETITION_PENALTY = 1.05
    GEN_LENGTH_PENALTY     = 1.1
    MAX_FINDINGS_TOKENS   = 192
    MAX_IMPRESSION_TOKENS = 128
    MAX_NODE_TEXT_TOKENS  = 32
    GRAD_ACCUM_STEPS = 1
else:
    raise ValueError(f"config.DATASET must be 'FUS' or 'CXR', got {DATASET!r}")

TRUST_REMOTE_CODE   = True
BACKBONE_DTYPE      = 'bfloat16'

HF_CACHE_DIR = os.environ.get('HF_HOME', os.path.abspath(
    os.path.join(os.path.dirname(top_data_dir), 'hf_cache')
))
os.makedirs(HF_CACHE_DIR, exist_ok=True)

if os.path.isdir(BACKBONE_MODEL_NAME):
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    print(f'[config.py] 使用本地 backbone 目录 -> {BACKBONE_MODEL_NAME}', flush=True)
else:
    _model_cache_dir = os.path.join(
        HF_CACHE_DIR, 'models--' + BACKBONE_MODEL_NAME.replace('/', '--'),
    )
    if os.path.exists(_model_cache_dir):
        os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
        if _looks_like_plain_hf_model_dir(_model_cache_dir):
            BACKBONE_MODEL_NAME = _model_cache_dir
            print(f'[config.py] 使用 ModelScope/plain-local backbone -> {_model_cache_dir}',
                  flush=True)
        else:
            print(f'[config.py] HF cache 已存在, 启用 offline 模式 -> {_model_cache_dir}',
                  flush=True)
    else:
        print(f'[config.py] HF cache 不存在, 首次启动将下载 (~8GB) -> {_model_cache_dir}',
              flush=True)
    del _model_cache_dir

from huggingface_hub import constants as _hf_constants
_hf_constants.ENDPOINT = _HF_MIRROR
_hf_constants.HUGGINGFACE_CO_URL_TEMPLATE = _HF_MIRROR + '/{repo_id}/resolve/{revision}/{filename}'
if hasattr(_hf_constants, 'HF_HUB_DISABLE_XET'):
    _hf_constants.HF_HUB_DISABLE_XET = True

train_data_name='train_data'
valid_data_name='valid_data'
test_data_name='test_data'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


raw_data_dir=os.path.join(top_data_dir,'raw_data/')
train_raw_data_path=os.path.join(raw_data_dir,'{}.jsonl'.format(train_data_name))
valid_raw_data_path=os.path.join(raw_data_dir,'{}.jsonl'.format(valid_data_name))
test_raw_data_path=os.path.join(raw_data_dir,'{}.jsonl'.format(test_data_name))

raw_hclg_dir=os.path.join(top_data_dir,'raw_hclg/')
train_raw_hclg_path=os.path.join(raw_hclg_dir,'{}.jsonl'.format(train_data_name))
valid_raw_hclg_path=os.path.join(raw_hclg_dir,'{}.jsonl'.format(valid_data_name))
test_raw_hclg_path=os.path.join(raw_hclg_dir,'{}.jsonl'.format(test_data_name))

max_len=None

avail_data_dir=os.path.join(top_data_dir,'avail_data/')
train_avail_data_path=os.path.join(avail_data_dir,'{}.pkl'.format(train_data_name))
valid_avail_data_path=os.path.join(avail_data_dir,'{}.pkl'.format(valid_data_name))
test_avail_data_path=os.path.join(avail_data_dir,'{}.pkl'.format(test_data_name))

HCLG_LLM_BACKEND        = 'vllm'

HCLG_LLM_URL   = "http://127.0.0.1:8000/v1"
HCLG_LLM_MODEL = "Qwen3-14B"
HCLG_LLM_KEY   = "EMPTY"

HCLG_LLM_REPO           = 'Qwen/Qwen3-14B'
HCLG_LLM_MODELSCOPE     = 'Qwen/Qwen3-14B'
HCLG_LLM_DIR            = os.path.join(HF_CACHE_DIR, 'hclg_llm', 'Qwen3-14B')
HCLG_LLM_TP             = 2
HCLG_LLM_DTYPE          = 'bfloat16'
HCLG_LLM_ENFORCE_EAGER  = True
HCLG_LLM_MAX_MODEL_LEN  = 16384
HCLG_LLM_GPU_MEM_UTIL   = 0.9

CHEXBERT_DIR = os.path.join(HF_CACHE_DIR, 'chexbert')
CHEXBERT_CKPT_PATH = os.path.join(CHEXBERT_DIR, 'chexbert.pth')

BERTSCORE_REPO = 'hfl/chinese-roberta-wwm-ext'
BERTSCORE_MODEL = os.path.join(HF_CACHE_DIR, 'bertscore/chinese-roberta-wwm-ext')
BERTSCORE_NUM_LAYERS = 8

RADGRAPH_DIR = os.path.join(HF_CACHE_DIR, 'radgraph')

LSC_LLM_REPO       = 'deepseek-ai/DeepSeek-V2-Lite-Chat'
LSC_LLM_MODELSCOPE = 'deepseek-ai/DeepSeek-V2-Lite-Chat'
LSC_LLM_DIR   = os.path.join(HF_CACHE_DIR, 'lsc_llm', 'DeepSeek-V2-Lite-Chat')
LSC_LLM_TP    = 2
LSC_LLM_DTYPE = 'bfloat16'
LSC_LLM_ENFORCE_EAGER = True

JUDGE_LLM_REPO       = 'hugging-quants/Meta-Llama-3.3-70B-Instruct-AWQ-INT4'
JUDGE_LLM_MODELSCOPE = 'LLM-Research/Meta-Llama-3.3-70B-Instruct-AWQ-INT4'
JUDGE_LLM_DIR        = os.path.join(HF_CACHE_DIR, 'judge_llm', 'Meta-Llama-3.3-70B-Instruct-AWQ-INT4')
JUDGE_LLM_TP         = 2
JUDGE_LLM_DTYPE      = 'float16'
JUDGE_LLM_QUANTIZATION = 'awq_marlin'
JUDGE_LLM_ENFORCE_EAGER = False
JUDGE_LLM_MAX_MODEL_LEN = 8192
JUDGE_LLM_GPU_MEM_UTIL  = 0.85



node_emb_cache_path = os.path.join(avail_data_dir, 'node_text_embeddings.pt')

model_dir  = os.path.join(top_data_dir, 'model/')
result_dir = os.path.join(top_data_dir, 'result/')

MODEL_NAME    = f'HCLGAlign_{MODEL_VERSION}'

model_save_dir  = os.path.join(model_dir, MODEL_NAME)
best_ckpt_dir   = os.path.join(model_save_dir, 'best')
latest_ckpt_dir = os.path.join(model_save_dir, 'latest')
train_log_path  = os.path.join(model_save_dir, 'train.log')


result_save_dir   = os.path.join(result_dir, MODEL_NAME)
test_pred_path    = os.path.join(result_save_dir, 'test_predictions.jsonl')
val_pred_path     = os.path.join(result_save_dir, 'val_predictions.jsonl')
eval_metrics_path = os.path.join(result_save_dir, 'eval_metrics.json')




EDGE_TYPES = [
    'has_obs',
    'has_obs_rev',
    'co_diagnostic',
    'conditional',
    'conditional_rev',
    'discrepancy',
]
EDGE_TYPE_TO_ID = {name: i for i, name in enumerate(EDGE_TYPES)}
NUM_RELATIONS   = len(EDGE_TYPES)
ASYMMETRIC_RELS = {'has_obs', 'conditional'}
SYMMETRIC_RELS  = {'co_diagnostic', 'discrepancy'}

NODE_TYPE_TO_ID = {'entity': 0, 'observation': 1}
NUM_NODE_TYPES  = 2

K_NEG          = 3

EDGE_DROPOUT_TARGET_REL = 'co_diagnostic'

USE_LOG_TRANSFORM = True

IGNORE_INDEX = -100

NUM_WORKERS      = 4
PIN_MEMORY       = True

USE_GRADIENT_CHECKPOINTING = True



D_INIT = None
D_LLM  = None

D_G = 512


NUM_MLP_HIDDEN = 256


USE_NODE_TYPE_EMB = False


RGCN_NUM_LAYERS = 3

RGCN_NUM_BASES = 4

RGCN_DROPOUT = 0.2


PROJ_HIDDEN  = None
PROJ_DROPOUT = 0.1


LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                       'gate_proj', 'up_proj', 'down_proj']


INFONCE_TAU_INIT      = 0.07

INFONCE_TAU_LEARNABLE = True

LAMBDA_INIT          = 1.0
LAMBDA_FINAL         = 0.1
LAMBDA_PHASE1_RATIO  = 0.3


GRAPH_DROPOUT_P = 0.1


USE_PREFIX_TUNING       = True
PREFIX_NUM_VIRTUAL_TOKENS = 10
PREFIX_PROJ_DROPOUT     = 0.1


GRAPH_FUSION_MODE      = 'crossattn'
CROSS_ATTN_EVERY       = 4
CROSS_ATTN_NUM_HEADS   = 8
CROSS_ATTN_DROPOUT     = 0.0


LR              = 5e-4
LR_GNN          = 5e-4
WEIGHT_DECAY    = 0.01

WARMUP_RATIO    = 0.05
MAX_GRAD_NORM   = 1.0



GEN_DO_SAMPLE         = False



TTC_K = 10

TTC_SAMPLING_CONFIGS = [
    {'do_sample': True,  'temperature': 0.5, 'top_p': 0.9},
    {'do_sample': True,  'temperature': 0.7, 'top_p': 0.9},
    {'do_sample': True,  'temperature': 0.9, 'top_p': 0.95},
    {'do_sample': True,  'temperature': 1.0, 'top_p': 0.95},
    {'do_sample': False, 'num_beams': 4},
    {'do_sample': False, 'num_beams': 6},
]

TTC_LENGTH_WEIGHT = 0.5

TTC_LENGTH_SIGMA = 0.5

TTC_TARGET_LENGTH_AUTO = True
TTC_TARGET_LENGTH_FALLBACK = 30.0


TTC_MBR_WEIGHT = 0.0

import random
import torch
import numpy as np
def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic = True
seeds=[0,42,7,23,124,1084,87]
seed_torch(seeds[1])



