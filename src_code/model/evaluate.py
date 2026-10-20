import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import config

import jieba
from rouge_score import rouge_scorer


class _ChineseTokenizer:

    def tokenize(self, text: str) -> List[str]:
        return [t for t in jieba.lcut(text) if t.strip()]


_ROUGE_KEYS = ('rouge1', 'rouge2', 'rougeLsum')
_OUT_PREFIX = {'rouge1': 'rouge_1', 'rouge2': 'rouge_2',
               'rougeLsum': 'rouge_l'}


def _split_sentences_for_rouge_l(text: str, language: str) -> str:

    import re
    if language == 'zh':
        return re.sub(r'(?<=[。；！？])\s*', '\n', text)
    return re.sub(r'(?<=[.;!?])\s+', '\n', text.strip())


def compute_rouge(predictions: List[str],
                  references: List[str],
                  language: str = 'zh') -> Dict[str, float]:
    assert len(predictions) == len(references), \
        f'predictions ({len(predictions)}) != references ({len(references)})'

    if language == 'zh':
        scorer = rouge_scorer.RougeScorer(
            list(_ROUGE_KEYS), use_stemmer=False, tokenizer=_ChineseTokenizer()
        )
    else:
        scorer = rouge_scorer.RougeScorer(list(_ROUGE_KEYS), use_stemmer=True)

    sums: Dict[str, List[float]] = {k: [] for k in _ROUGE_KEYS}
    sums_p: Dict[str, List[float]] = {k: [] for k in _ROUGE_KEYS}
    sums_r: Dict[str, List[float]] = {k: [] for k in _ROUGE_KEYS}

    for pred, ref in zip(predictions, references):
        ref_s  = _split_sentences_for_rouge_l(ref,  language)
        pred_s = _split_sentences_for_rouge_l(pred, language)
        result = scorer.score(ref_s, pred_s)
        for k in _ROUGE_KEYS:
            sums_p[k].append(result[k].precision)
            sums_r[k].append(result[k].recall)
            sums[k].append(result[k].fmeasure)

    n = len(predictions)
    if n == 0:
        out = {'n_samples': 0}
        for k in _ROUGE_KEYS:
            p = _OUT_PREFIX[k]
            out[f'{p}_p'] = out[f'{p}_r'] = out[f'{p}_f'] = 0.0
        return out

    out: Dict[str, float] = {'n_samples': n}
    for k in _ROUGE_KEYS:
        p = _OUT_PREFIX[k]
        out[f'{p}_p'] = sum(sums_p[k]) / n
        out[f'{p}_r'] = sum(sums_r[k]) / n
        out[f'{p}_f'] = sum(sums[k]) / n
    return out


def compute_rouge_l(predictions: List[str],
                     references: List[str],
                     language: str = 'zh') -> Dict[str, float]:

    return compute_rouge(predictions, references, language=language)


def _hf_weight_ignore(repo: str) -> List[str]:

    ignore = ['*.h5', '*.msgpack', '*.ot', 'tf_model*', 'flax_model*']
    try:
        from huggingface_hub import list_repo_files
        files = list_repo_files(repo)
        if any(f.endswith('.safetensors') for f in files):
            ignore += ['*.bin', '*.pth', '*.pt', 'consolidated*', 'pytorch_model*']
    except Exception as e:
        logging.warning('list_repo_files(%s) 失败, 不去重权重格式 (可能多下一份): %s', repo, e)
    return ignore


def _dir_has_weights(model_dir: str) -> bool:

    if not (os.path.isdir(model_dir) and os.path.exists(os.path.join(model_dir, 'config.json'))):
        return False
    for name in os.listdir(model_dir):
        if name.endswith(('.safetensors', '.bin', '.gguf', '.pt')):
            return True
    return False


def _modelscope_download(ms_id: str, model_dir: str) -> bool:

    try:
        from modelscope import snapshot_download as ms_snapshot
    except ImportError:
        logging.info('未装 modelscope, 跳过国内 CDN 下载 (pip install modelscope 可启用)')
        return False
    try:
        os.makedirs(model_dir, exist_ok=True)
        logging.info('ModelScope 下载 LSC 抽取模型 %s -> %s (国内 CDN, 绕开 HF Xet)', ms_id, model_dir)
        ms_snapshot(model_id=ms_id, local_dir=model_dir,
                    ignore_file_pattern=[r'.*\.h5$', r'.*\.msgpack$', r'.*\.ot$',
                                         r'.*\.bin$', r'.*\.pth$', r'.*consolidated.*'])
        return _dir_has_weights(model_dir)
    except Exception as e:
        logging.warning('ModelScope 下载失败 (%s), 退回 HF: %s', ms_id, e)
        return False


def _patch_bertscore_tokenizer_maxlen(model_dir: str, max_len: int = 512) -> None:

    cfg_path = os.path.join(model_dir, 'tokenizer_config.json')
    try:
        cfg = {}
        if os.path.exists(cfg_path):
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
        cur = cfg.get('model_max_length')
        if not isinstance(cur, int) or cur > 1_000_000:
            cfg['model_max_length'] = max_len
            with open(cfg_path, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            logging.info('已给 %s 设 model_max_length=%d (修复 bert_score 溢出)', cfg_path, max_len)
    except Exception as e:
        logging.warning('patch tokenizer_config model_max_length 失败 (忽略): %s', e)


def _ensure_bertscore_model(model_dir: str, repo: str) -> str:

    if not (os.path.isdir(model_dir) and os.path.exists(os.path.join(model_dir, 'config.json'))):
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise ImportError('需要 huggingface_hub 才能下载 BERTScore 模型') from e
        os.makedirs(model_dir, exist_ok=True)
        snapshot_download(repo_id=repo, local_dir=model_dir,
                          ignore_patterns=_hf_weight_ignore(repo))
    _patch_bertscore_tokenizer_maxlen(model_dir)
    return model_dir


def compute_bertscore(predictions: List[str],
                      references: List[str],
                      model_type: Optional[str] = None,
                      num_layers: Optional[int] = None,
                      lang: str = 'zh',
                      device: Optional[str] = None,
                      batch_size: int = 64) -> Dict[str, float]:

    assert len(predictions) == len(references), \
        f'predictions ({len(predictions)}) != references ({len(references)})'

    n = len(predictions)
    if n == 0:
        return {'bertscore_p': 0.0, 'bertscore_r': 0.0, 'bertscore_f': 0.0, 'n_samples': 0}

    _disable_hf_offline_for_download()

    try:
        from bert_score import score as _bertscore
    except ImportError as e:
        raise ImportError(
            'compute_bertscore 需要 bert-score: `pip install bert-score`'
        ) from e

    if model_type is None:
        if lang == 'en':
            model_type = 'roberta-large'
            if num_layers is None:
                num_layers = 17
        else:
            model_type = getattr(config, 'BERTSCORE_MODEL', 'hfl/chinese-roberta-wwm-ext')
    if num_layers is None:
        num_layers = getattr(config, 'BERTSCORE_NUM_LAYERS', 8)

    if model_type.startswith(('.', '/', '~')) or model_type.count('/') >= 2:
        repo = getattr(config, 'BERTSCORE_REPO', 'hfl/chinese-roberta-wwm-ext')
        model_type = _ensure_bertscore_model(model_type, repo)

    P, R, F = _bertscore(
        predictions, references,
        model_type=model_type, num_layers=num_layers, lang=lang,
        rescale_with_baseline=False,
        verbose=False, device=device, batch_size=batch_size,
    )
    return {
        'bertscore_p': float(P.mean().item()),
        'bertscore_r': float(R.mean().item()),
        'bertscore_f': float(F.mean().item()),
        'n_samples': n,
    }



LSC_SYSTEM_PROMPT = (
    '你是一个极其严谨的超声科专家。你的任务是从【超声诊断结论(Impression)】中，'
    '拆解并提取出所有独立的定性诊断、解剖特征和临床状态，形成一个标准化的列表。\n'
    '提取规则：\n'
    '1. 只需提取诊断或临床状态本身（如：“右侧主支气管闭锁或梗阻可能”、“左肺受压”、“单胎”、“头位”）。\n'
    '2. 保持医学术语的精简，去除口语化修饰词和多余的连词。\n'
    '3. 当结论中没有可提取的诊断或临床状态时，请返回空列表。'
)
_LSC_JSON_HINT = (
    '\n\n请严格以 JSON 对象输出，且只输出 JSON，不要任何额外文字：'
    '{"statuses": ["状态1", "状态2", ...]}。若无异常状态则返回 {"statuses": []}。'
)
LSC_USER_TEMPLATE = '请提取以下结论：\n{impression_text}'


def _normalize_state(s: str) -> str:

    s = s.strip().lower()
    return s.strip('。，、；：,.;: \t\n（）()【】[]"\'')


def _jieba_token_set(s: str) -> Set[str]:
    return {t for t in jieba.lcut(s) if t.strip()}


def _states_match(a: str, b: str, mode: str) -> bool:

    if not a or not b:
        return a == b
    if mode == 'substring':
        return (a in b) or (b in a)
    if mode == 'jieba_subset':
        ta, tb = _jieba_token_set(a), _jieba_token_set(b)
        if not ta or not tb:
            return a == b
        return ta.issubset(tb) or tb.issubset(ta)
    raise ValueError(f'未知 match_mode: {mode!r} (应为 substring / jieba_subset)')


def _maximum_one_to_one_matches(gen_set: Set[str],
                                gt_set: Set[str],
                                match_mode: str) -> int:
    gen_states = sorted(gen_set)
    gt_states = sorted(gt_set)
    adjacency = [
        [j for j, gt in enumerate(gt_states) if _states_match(gen, gt, match_mode)]
        for gen in gen_states
    ]
    gt_to_gen: Dict[int, int] = {}

    def augment(gen_idx: int, visited_gt: Set[int]) -> bool:
        for gt_idx in adjacency[gen_idx]:
            if gt_idx in visited_gt:
                continue
            visited_gt.add(gt_idx)
            previous_gen = gt_to_gen.get(gt_idx)
            if previous_gen is None or augment(previous_gen, visited_gt):
                gt_to_gen[gt_idx] = gen_idx
                return True
        return False

    return sum(augment(i, set()) for i in range(len(gen_states)))


def lsc_soft_match_prf(gen_set: Set[str],
                       gt_set: Set[str],
                       match_mode: str = 'jieba_subset') -> Dict[str, float]:
    if not gen_set and not gt_set:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0,
                'tp': 0, 'n_gen': 0, 'n_gt': 0}

    tp = _maximum_one_to_one_matches(gen_set, gt_set, match_mode)

    n_gen, n_gt = len(gen_set), len(gt_set)
    precision = tp / n_gen if n_gen > 0 else 0.0
    recall = tp / n_gt if n_gt > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {'precision': precision, 'recall': recall, 'f1': f1,
            'tp': tp, 'n_gen': n_gen, 'n_gt': n_gt}


def _build_status_schema():
    """懒构造 pydantic schema (避免 import evaluate.py 就强依赖 pydantic)."""
    from pydantic import BaseModel, Field

    class ClinicalStatusExtraction(BaseModel):
        statuses: List[str] = Field(
            default_factory=list,
            description=('从超声结论中提取的独立定性诊断、解剖异常或病理状态列表。'
                         "例如：['右侧主支气管闭锁可能', '心胸比例偏小', '羊水过多']"),
        )

    return ClinicalStatusExtraction


class LSCEvaluator:

    def __init__(self,
                 backend: str = 'vllm',
                 model_dir: Optional[str] = None,
                 repo: Optional[str] = None,
                 modelscope_id: Optional[str] = None,
                 tensor_parallel_size: Optional[int] = None,
                 dtype: Optional[str] = None,
                 enforce_eager: Optional[bool] = None,
                 max_model_len: int = 8192,
                 gpu_memory_utilization: float = 0.90,
                 openai_model: str = 'gpt-4o-2024-08-06',
                 temperature: float = 0.0,
                 seed: int = 42,
                 max_retries: int = 3,
                 timeout: float = 90.0,
                 match_mode: str = 'jieba_subset',
                 cache_path: Optional[str] = None):
        if backend in ('auto', 'local'):
            backend = 'vllm'
        if backend not in ('vllm', 'openai'):
            raise ValueError(f'未知 backend: {backend!r} (应为 vllm/openai/auto)')
        self.backend = backend
        self.match_mode = match_mode
        self.temperature = temperature
        self.seed = seed
        self.max_retries = max_retries
        self._schema = _build_status_schema()

        if backend == 'vllm':
            self.model_dir = model_dir or getattr(config, 'LSC_LLM_DIR', None)
            self.repo = repo or getattr(config, 'LSC_LLM_REPO', None)
            self.ms_id = modelscope_id or getattr(config, 'LSC_LLM_MODELSCOPE', None)
            if not self.model_dir:
                raise ValueError('vllm 后端需要 config.LSC_LLM_DIR 或 model_dir')
            self.tp = tensor_parallel_size or getattr(config, 'LSC_LLM_TP', 1)
            self.dtype = dtype or getattr(config, 'LSC_LLM_DTYPE', 'bfloat16')
            self.enforce_eager = (enforce_eager if enforce_eager is not None
                                  else getattr(config, 'LSC_LLM_ENFORCE_EAGER', False))
            self.max_model_len = max_model_len
            self.gpu_memory_utilization = gpu_memory_utilization
            self.model = os.path.basename(str(self.model_dir).rstrip('/\\')) or 'lsc_llm'
            self._llm = None
            self._sampling = None
        else:
            self.model = openai_model
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError('openai 后端需要: `pip install openai`') from e
            self.client = OpenAI(timeout=timeout)

        self.cache_path = cache_path
        self._cache: Dict[str, List[str]] = {}
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    self._cache = json.load(f)
                logging.info('LSC 抽取缓存载入 %d 条 <- %s', len(self._cache), cache_path)
            except Exception as e:
                logging.warning('LSC 缓存读取失败 (%s), 忽略并重建: %s', cache_path, e)
                self._cache = {}

    def _cache_key(self, text: str) -> str:
        return f'{self.model}\x00{text}'

    def _save_cache(self) -> None:
        if not self.cache_path:
            return
        try:
            os.makedirs(os.path.dirname(self.cache_path) or '.', exist_ok=True)
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f, ensure_ascii=False)
        except Exception as e:
            logging.warning('LSC 缓存落盘失败: %s', e)

    @staticmethod
    def _parse_statuses(content: str) -> List[str]:
        """防御式解析 LLM 吐的 JSON -> 状态字符串列表. 解析不出来就返回 []."""
        content = (content or '').strip()
        if not content:
            return []
        if content.startswith('```'):
            import re
            content = re.sub(r'^```(?:json)?\s*|\s*```$', '', content, flags=re.MULTILINE).strip()
        parsed = None
        try:
            parsed = json.loads(content)
        except Exception:
            import re
            m = re.search(r'\{.*\}', content, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except Exception:
                    parsed = None
        if isinstance(parsed, dict):
            statuses = parsed.get('statuses', [])
        elif isinstance(parsed, list):
            statuses = parsed
        else:
            return []
        return [str(s) for s in statuses] if isinstance(statuses, list) else []

    def _messages(self, text: str) -> List[Dict[str, str]]:
        return [
            {'role': 'system', 'content': LSC_SYSTEM_PROMPT + _LSC_JSON_HINT},
            {'role': 'user', 'content': LSC_USER_TEMPLATE.format(impression_text=text)},
        ]

    def _ensure_vllm(self) -> None:
        if self._llm is not None:
            return
        if _dir_has_weights(self.model_dir):
            logging.info('LSC 抽取模型本地已就位, 跳过下载: %s', self.model_dir)
        else:
            _disable_hf_offline_for_download()
            os.makedirs(self.model_dir, exist_ok=True)
            ok = False
            if self.ms_id:
                ok = _modelscope_download(self.ms_id, self.model_dir)
            if not ok and self.repo:
                try:
                    from huggingface_hub import snapshot_download
                    logging.info('HF 下载 LSC 抽取模型 %s -> %s', self.repo, self.model_dir)
                    snapshot_download(repo_id=self.repo, local_dir=self.model_dir,
                                      ignore_patterns=_hf_weight_ignore(self.repo))
                    ok = _dir_has_weights(self.model_dir)
                except Exception as e:
                    logging.warning('HF snapshot_download 失败: %s', e)
            if not _dir_has_weights(self.model_dir):
                raise FileNotFoundError(
                    f'{self.model_dir} 无可用模型权重. 墙内请优先用 ModelScope 预下载: '
                    f'`modelscope download --model {self.ms_id or "<id>"} '
                    f'--local_dir {self.model_dir}` (HF 热门仓多为 Xet 存储, hf-mirror 不代理 '
                    f'cas-bridge.xethub.hf.co, 墙内会超时).')
        os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
        if self.enforce_eager:
            os.environ.setdefault('VLLM_USE_V1', '0')
            os.environ.setdefault('VLLM_ATTENTION_BACKEND', 'XFORMERS')
        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError(
                f'导入 vllm 失败: {type(e).__name__}: {e}  '
                f'(若已 pip install vllm, 多半是环境不对 / vllm 与 torch-CUDA 版本不匹配; '
                f'先在该环境跑 `python -c "import vllm; print(vllm.__version__)"` 看真实报错)'
            ) from e
        logging.info('vLLM 加载 LSC 抽取模型 (tp=%d, dtype=%s, eager=%s) ...',
                     self.tp, self.dtype, self.enforce_eager)
        self._llm = LLM(
            model=self.model_dir, tokenizer=self.model_dir,
            tensor_parallel_size=self.tp, dtype=self.dtype,
            trust_remote_code=True,
            tokenizer_mode='slow',
            enforce_eager=self.enforce_eager,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len, seed=self.seed,
        )
        self._sampling = SamplingParams(temperature=self.temperature, max_tokens=512, seed=self.seed)

    def _extract_vllm_batch(self, texts: List[str]) -> List[List[str]]:
        self._ensure_vllm()
        convs = [self._messages(t) for t in texts]
        outputs = self._llm.chat(convs, self._sampling, use_tqdm=True)
        return [self._parse_statuses(o.outputs[0].text) for o in outputs]

    def _extract_openai_one(self, text: str) -> List[str]:
        last_err: Optional[str] = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[
                        {'role': 'system', 'content': LSC_SYSTEM_PROMPT},
                        {'role': 'user', 'content': LSC_USER_TEMPLATE.format(impression_text=text)},
                    ],
                    response_format=self._schema,
                    temperature=self.temperature, seed=self.seed,
                )
                msg = resp.choices[0].message
                if getattr(msg, 'refusal', None):
                    raise RuntimeError(f'OpenAI 拒答: {msg.refusal}')
                return list(msg.parsed.statuses) if msg.parsed else []
            except Exception as e:
                last_err = f'{type(e).__name__}: {e}'
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
        logging.warning('LSC OpenAI 抽取失败 (%d 次), 按空集处理: %s | %.40s',
                        self.max_retries, last_err, text.replace('\n', ' '))
        return []

    def extract_raw_many(self, texts: List[str]) -> Dict[str, List[str]]:
        uniq = list(dict.fromkeys(texts))
        out: Dict[str, List[str]] = {}
        todo: List[str] = []
        for t in uniq:
            if not t or not t.strip():
                out[t] = []
                continue
            key = self._cache_key(t)
            if key in self._cache:
                out[t] = self._cache[key]
            else:
                todo.append(t)

        if todo:
            if self.backend == 'vllm':
                raws = self._extract_vllm_batch(todo)
            else:
                raws = [self._extract_openai_one(t) for t in todo]
            for t, raw in zip(todo, raws):
                self._cache[self._cache_key(t)] = raw
                out[t] = raw
            self._save_cache()
        return out

    def _to_set(self, raw: List[str]) -> Set[str]:
        return {_normalize_state(s) for s in raw if s and s.strip()}

    def extract_statuses(self, text: str) -> Set[str]:
        if not text or not text.strip():
            return set()
        return self._to_set(self.extract_raw_many([text]).get(text, []))

    def score_pair(self, generated: str, ground_truth: str) -> Dict[str, Any]:
        raw = self.extract_raw_many([generated, ground_truth])
        gen_set = self._to_set(raw.get(generated, []))
        gt_set = self._to_set(raw.get(ground_truth, []))
        prf = lsc_soft_match_prf(gen_set, gt_set, match_mode=self.match_mode)
        prf['gen_states'] = sorted(gen_set)
        prf['gt_states'] = sorted(gt_set)
        return prf

    def evaluate(self,
                 predictions: List[str],
                 references: List[str],
                 verbose: bool = False) -> Dict[str, float]:
        assert len(predictions) == len(references), \
            f'predictions ({len(predictions)}) != references ({len(references)})'

        n = len(predictions)
        if n == 0:
            return {'lsc_f1': 0.0, 'lsc_precision': 0.0, 'lsc_recall': 0.0,
                    'lsc_f1_micro': 0.0, 'n_samples': 0}

        raw = self.extract_raw_many(list(predictions) + list(references))

        p_list, r_list, f_list = [], [], []
        tot_tp = tot_gen = tot_gt = 0
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            gen_set = self._to_set(raw.get(pred, []))
            gt_set = self._to_set(raw.get(ref, []))
            prf = lsc_soft_match_prf(gen_set, gt_set, match_mode=self.match_mode)
            p_list.append(prf['precision'])
            r_list.append(prf['recall'])
            f_list.append(prf['f1'])
            tot_tp += prf['tp']
            tot_gen += prf['n_gen']
            tot_gt += prf['n_gt']
            if verbose:
                logging.info('[LSC %d/%d] P=%.3f R=%.3f F1=%.3f | gen=%s | gt=%s',
                             i + 1, n, prf['precision'], prf['recall'], prf['f1'],
                             sorted(gen_set), sorted(gt_set))

        micro_p = tot_tp / tot_gen if tot_gen > 0 else 0.0
        micro_r = tot_tp / tot_gt if tot_gt > 0 else 0.0
        micro_f = (2 * micro_p * micro_r / (micro_p + micro_r)
                   if (micro_p + micro_r) > 0 else 0.0)
        return {
            'lsc_f1': sum(f_list) / n,
            'lsc_precision': sum(p_list) / n,
            'lsc_recall': sum(r_list) / n,
            'lsc_f1_micro': micro_f,
            'n_samples': n,
        }


def compute_lsc_f1(predictions: List[str],
                   references: List[str],
                   backend: str = 'local',
                   match_mode: str = 'jieba_subset',
                   cache_path: Optional[str] = None,
                   **kwargs: Any) -> Dict[str, float]:
    evaluator = LSCEvaluator(backend=backend, match_mode=match_mode,
                             cache_path=cache_path, **kwargs)
    return evaluator.evaluate(predictions, references)



_CHEXBERT_HF_REPO = 'StanfordAIMI/RRG_scorers'
_CHEXBERT_CKPT_FILENAME = 'chexbert.pth'

_CHEXBERT_TARGET_NAMES = [
    'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema',
    'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices', 'No Finding',
]
_CHEXBERT_5_NAMES = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']
_CHEXBERT_5_INDEX = [_CHEXBERT_TARGET_NAMES.index(x) for x in _CHEXBERT_5_NAMES]


def _download_chexbert_to(dest_dir: str) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError('需要 huggingface_hub 才能下载 chexbert.pth') from e
    os.makedirs(dest_dir, exist_ok=True)
    return hf_hub_download(repo_id=_CHEXBERT_HF_REPO,
                           filename=_CHEXBERT_CKPT_FILENAME,
                           local_dir=dest_dir)


def _ensure_chexbert_checkpoint(checkpoint_path: Optional[str] = None) -> str:
    if checkpoint_path is None:
        checkpoint_path = getattr(config, 'CHEXBERT_CKPT_PATH', None)
    if not checkpoint_path:
        raise ValueError('未提供 checkpoint_path 且 config.CHEXBERT_CKPT_PATH 未配置')

    src = os.path.abspath(os.path.expanduser(checkpoint_path))
    if not os.path.exists(src):
        _download_chexbert_to(os.path.dirname(src))
        if not os.path.exists(src):
            raise FileNotFoundError(f'下载后仍未在 {src} 找到 chexbert.pth')
    return src


def _disable_hf_offline_for_download() -> None:
    import importlib
    os.environ['HF_HUB_OFFLINE'] = '0'
    os.environ['TRANSFORMERS_OFFLINE'] = '0'
    for mod_name, attr in (
        ('huggingface_hub.constants', 'HF_HUB_OFFLINE'),
        ('huggingface_hub.file_download', 'HF_HUB_OFFLINE'),
        ('transformers.utils.hub', '_is_offline_mode'),
    ):
        try:
            m = importlib.import_module(mod_name)
            if hasattr(m, attr):
                setattr(m, attr, False)
        except Exception:
            pass


class _CheXbertScorer:

    def __init__(self, checkpoint: str, device: Optional[str] = None):
        import torch
        import torch.nn as nn
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        self._torch = torch
        self.device = torch.device(
            device or ('cuda' if torch.cuda.is_available() else 'cpu'))

        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        config = AutoConfig.from_pretrained('bert-base-uncased')
        hidden = config.hidden_size

        class _BertLabeler(nn.Module):
            def __init__(self):
                super().__init__()
                self.bert = AutoModel.from_config(config)
                self.dropout = nn.Dropout(0.1)
                self.linear_heads = nn.ModuleList(
                    [nn.Linear(hidden, 4) for _ in range(13)] + [nn.Linear(hidden, 2)])

            def forward(self, input_ids, attention_mask):
                hid = self.bert(input_ids, attention_mask=attention_mask)[0]
                cls = self.dropout(hid[:, 0, :])
                return [head(cls) for head in self.linear_heads]

        model = _BertLabeler()
        state = torch.load(checkpoint, map_location=self.device, weights_only=False)
        state = state.get('model_state_dict', state)
        clean = {k.replace('module.', ''): v for k, v in state.items()}
        model.load_state_dict(clean, strict=False)
        self.model = model.to(self.device).eval()

    def label(self, report: str, uncertain_handling: str = 'positive') -> List[int]:

        torch = self._torch
        enc = self.tokenizer(report or '', truncation=True, max_length=512,
                             return_tensors='pt')
        input_ids = enc['input_ids'].to(self.device)
        attn = enc['attention_mask'].to(self.device)
        with torch.no_grad():
            logits = self.model(input_ids, attn)
        cls_idx = [int(t.argmax(dim=1).item()) for t in logits]
        if uncertain_handling == 'negative':
            return [1 if c == 1 else 0 for c in cls_idx]
        return [1 if c in (1, 3) else 0 for c in cls_idx]


def compute_f1_chexbert(predictions: List[str],
                        references: List[str],
                        checkpoint_path: Optional[str] = None,
                        device: Optional[str] = None,
                        uncertain: str = 'negative',
                        aggregation: str = 'macro',
                        n_classes: int = 14) -> Dict[str, float]:

    assert len(predictions) == len(references), \
        f'predictions ({len(predictions)}) != references ({len(references)})'

    n = len(predictions)
    if n == 0:
        return {'f1_chexbert': 0.0, 'f1_chexbert_macro': 0.0,
                'f1_chexbert_5': 0.0, 'accuracy_5': 0.0, 'n_samples': 0}

    _disable_hf_offline_for_download()

    try:
        from sklearn.metrics import classification_report, accuracy_score
    except ImportError as e:
        raise ImportError(
            'compute_f1_chexbert 需要 scikit-learn: `pip install scikit-learn`'
        ) from e

    if uncertain not in ('positive', 'negative'):
        raise ValueError(f"uncertain 必须是 'positive' 或 'negative', 收到 {uncertain!r}")
    if aggregation not in ('micro', 'macro'):
        raise ValueError(f"aggregation 必须是 'micro' 或 'macro', 收到 {aggregation!r}")
    if n_classes not in (14, 5):
        raise ValueError(f"n_classes 必须是 14 或 5, 收到 {n_classes!r}")

    ckpt = _ensure_chexbert_checkpoint(checkpoint_path)
    scorer = _CheXbertScorer(ckpt, device=device)

    refs_lab = [scorer.label(r, uncertain_handling=uncertain) for r in references]
    hyps_lab = [scorer.label(p, uncertain_handling=uncertain) for p in predictions]

    cr_14 = classification_report(refs_lab, hyps_lab, target_names=_CHEXBERT_TARGET_NAMES,
                                  output_dict=True, zero_division=0)
    refs_5 = [[row[i] for i in _CHEXBERT_5_INDEX] for row in refs_lab]
    hyps_5 = [[row[i] for i in _CHEXBERT_5_INDEX] for row in hyps_lab]
    cr_5 = classification_report(refs_5, hyps_5, target_names=_CHEXBERT_5_NAMES,
                                 output_dict=True, zero_division=0)
    accuracy_5 = accuracy_score(refs_5, hyps_5)

    agg_key = 'micro avg' if aggregation == 'micro' else 'macro avg'
    cr_main = cr_14 if n_classes == 14 else cr_5
    main_f1 = float(cr_main[agg_key]['f1-score'])

    return {
        'f1_chexbert':            main_f1,
        '_uncertain':             uncertain,
        '_aggregation':           aggregation,
        '_n_classes':             n_classes,
        'f1_chexbert_micro_14':   float(cr_14['micro avg']['f1-score']),
        'f1_chexbert_macro_14':   float(cr_14['macro avg']['f1-score']),
        'f1_chexbert_micro_5':    float(cr_5['micro avg']['f1-score']),
        'f1_chexbert_macro_5':    float(cr_5['macro avg']['f1-score']),
        'f1_chexbert_macro':      float(cr_14['macro avg']['f1-score']),
        'f1_chexbert_5':          float(cr_5['micro avg']['f1-score']),
        'accuracy_5':             float(accuracy_5),
        'n_samples':              n,
    }




def _radgraph_shim_transformers_tokenizer() -> None:

    try:
        from transformers import PreTrainedTokenizerBase as B
    except Exception as e:
        logging.warning('补 transformers tokenizer 垫片失败 (忽略): %s', e)
        return

    def _cls_sep(self):
        cls = [self.cls_token_id] if getattr(self, 'cls_token_id', None) is not None else []
        sep = [self.sep_token_id] if getattr(self, 'sep_token_id', None) is not None else []
        return cls, sep

    def encode_plus(self, *a, **k):
        return self(*a, **k)

    def batch_encode_plus(self, batch, *a, **k):
        return self(batch, *a, **k)

    def build_inputs_with_special_tokens(self, ids0, ids1=None):
        cls, sep = _cls_sep(self)
        return cls + ids0 + sep if ids1 is None else cls + ids0 + sep + ids1 + sep

    def create_token_type_ids_from_sequences(self, ids0, ids1=None):
        cls, sep = _cls_sep(self)
        if ids1 is None:
            return [0] * len(cls + ids0 + sep)
        return [0] * len(cls + ids0 + sep) + [1] * len(ids1 + sep)

    def get_special_tokens_mask(self, ids0, ids1=None, already_has_special_tokens=False):
        if already_has_special_tokens:
            return [1 if t in (self.cls_token_id, self.sep_token_id) else 0 for t in ids0]
        if ids1 is None:
            return [1] + [0] * len(ids0) + [1]
        return [1] + [0] * len(ids0) + [1] + [0] * len(ids1) + [1]

    def num_special_tokens_to_add(self, pair=False):
        return 3 if pair else 2

    shims = {
        'encode_plus': encode_plus,
        'batch_encode_plus': batch_encode_plus,
        'build_inputs_with_special_tokens': build_inputs_with_special_tokens,
        'create_token_type_ids_from_sequences': create_token_type_ids_from_sequences,
        'get_special_tokens_mask': get_special_tokens_mask,
        'num_special_tokens_to_add': num_special_tokens_to_add,
    }
    added = []
    for name, fn in shims.items():
        if not hasattr(B, name):
            setattr(B, name, fn)
            added.append(name)
    if added:
        logging.info('已为 radgraph 补 transformers 5.x 缺失的 tokenizer 接口: %s', added)


def compute_f1_radgraph(predictions: List[str],
                        references: List[str],
                        reward_level: str = 'partial',
                        model_type: str = 'radgraph',
                        model_cache_dir: Optional[str] = None) -> Dict[str, float]:
    assert len(predictions) == len(references), \
        f'predictions ({len(predictions)}) != references ({len(references)})'

    n = len(predictions)
    if n == 0:
        return {'f1_radgraph': 0.0, 'n_samples': 0}

    _disable_hf_offline_for_download()

    try:
        from radgraph import F1RadGraph
    except ImportError as e:
        raise ImportError(
            'compute_f1_radgraph 需要 radgraph: `pip install radgraph`'
        ) from e

    _radgraph_shim_transformers_tokenizer()

    if model_cache_dir is None:
        model_cache_dir = getattr(config, 'RADGRAPH_DIR', None)
    rg_kwargs: Dict[str, Any] = {}
    if model_cache_dir:
        os.makedirs(model_cache_dir, exist_ok=True)
        rg_kwargs['model_cache_dir'] = model_cache_dir
        rg_kwargs['tokenizer_cache_dir'] = model_cache_dir

    f1radgraph = F1RadGraph(reward_level=reward_level, model_type=model_type, **rg_kwargs)
    mean_reward, _reward_list, _hyp_anns, _ref_anns = f1radgraph(hyps=predictions, refs=references)

    if reward_level == 'all':
        simple, partial, complete = mean_reward
        return {
            'f1_radgraph_simple': float(simple),
            'f1_radgraph_partial': float(partial),
            'f1_radgraph_complete': float(complete),
            'n_samples': n,
        }
    return {'f1_radgraph': float(mean_reward), 'n_samples': n}



JUDGE_SYSTEM_PROMPT_ZH = (
    "你是一位严谨且经验丰富的医学超声/放射科主任医师。你的任务是根据提供的原始【检查所见】, "
    "客观评价两个 AI 生成的【诊断结论(A和B)】。\n\n"
    "请严格按以下步骤进行临床逻辑核查 (Chain of Thought):\n"
    "1. **临床逻辑推演 (Clinical Deduction)**: 仔细阅读 Findings 中的所有定量测量数值"
    "(如径线大小、厚度、孕周等)和解剖形态描述。推演出基于这些数据, 最终的结论中应当"
    "包含哪些【定性诊断与病理状态】(例如: 看到'羊水暗区86mm', 应推演出'羊水过多'; "
    "看到'心胸比例0.39', 应推演出'心胸比例偏小')。\n"
    "2. **评价结论 A**: A 的定性诊断是否与 Findings 的底层数据逻辑完美契合? "
    "(重点核查是否发生了漏诊, 或产生了编造数据的幻觉/过度诊断)。\n"
    "3. **评价结论 B**: 同上, 指出在定性诊断和病理推演上的错误、幻觉或漏诊。\n"
    "4. **最终判决**: 比较 A 和 B, 哪一个在'从定量到定性的医学推理准确度'和'临床事实无误'上"
    "表现更好? (如果不相上下, 或均存在不可接受的致命临床错误, 请判 Tie)。\n\n"
    "请严格以 JSON 对象输出, 且只输出 JSON, 不要任何额外文字:\n"
    '{"clinical_deduction":"<推演内容>","reasoning_A":"<评 A>","reasoning_B":"<评 B>",'
    '"final_decision":"A"}\n'
    'final_decision 必须严格是 "A" / "B" / "Tie" 三者之一。'
)
JUDGE_USER_TEMPLATE_ZH = (
    "请评价以下临床病例:\n\n"
    "【检查所见】\n{findings}\n\n"
    "【结论 A】\n{impression_A}\n\n"
    "【结论 B】\n{impression_B}"
)

JUDGE_SYSTEM_PROMPT_EN = (
    "You are a rigorous and experienced Chief Physician of Medical Ultrasound/Radiology. "
    "Your task is to objectively evaluate two AI-generated Diagnostic Impressions (A and B) "
    "based on the provided original Findings.\n\n"
    "Please strictly follow these steps for clinical logical verification (Chain of Thought):\n"
    "1. **Clinical Deduction**: Carefully read all quantitative measurements and anatomical "
    "descriptions in the Findings. Deduce which [Qualitative Diagnoses and Pathological States] "
    'MUST be included (e.g., "AFV 86mm" implies "Polyhydramnios"; "Cardiothoracic ratio 0.39" '
    'implies "Small cardiothoracic ratio").\n'
    "2. **Evaluate Impression A**: Does the qualitative diagnosis in A perfectly align with the "
    "underlying data logic of the Findings? (Check for missed diagnoses or "
    "hallucinations/over-diagnoses).\n"
    "3. **Evaluate Impression B**: Similarly, point out any errors, hallucinations, or omissions "
    "in B regarding qualitative diagnosis and pathological deduction.\n"
    "4. **Final Verdict**: Compare A and B. Which one performs better in "
    '"medical reasoning accuracy from quantitative to qualitative" and "clinical fact correctness"? '
    '(If they are comparable or both contain unacceptable fatal clinical errors, output "Tie").\n\n'
    "Output strictly in JSON, nothing else:\n"
    '{"clinical_deduction":"...","reasoning_A":"...","reasoning_B":"...","final_decision":"A"}\n'
    'final_decision MUST be exactly one of "A" / "B" / "Tie".'
)
JUDGE_USER_TEMPLATE_EN = (
    "Please evaluate the following clinical case:\n\n"
    "[Findings]\n{findings}\n\n"
    "[Impression A]\n{impression_A}\n\n"
    "[Impression B]\n{impression_B}"
)


def _normalize_decision(d: Any) -> str:
    if not isinstance(d, str):
        return 'Tie'
    s = d.strip().upper()
    if s in ('A', 'B'):
        return s
    if s == 'TIE':
        return 'Tie'
    return 'Tie'


def _parse_judge_json(text: str) -> Dict[str, Any]:
    s = text.strip()
    if s.startswith('```'):
        import re as _re
        s = _re.sub(r'^```(?:json)?\s*|\s*```$', '', s, flags=_re.MULTILINE).strip()
    try:
        obj = json.loads(s)
    except Exception:
        obj = None
    if obj is None:
        import re as _re
        m = _re.search(r'\{.*\}', text, _re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = None
    if not isinstance(obj, dict):
        return {'clinical_deduction': '', 'reasoning_A': '', 'reasoning_B': '',
                'final_decision': 'Tie', '_parse_error': True}
    return {
        'clinical_deduction': str(obj.get('clinical_deduction', '')),
        'reasoning_A':        str(obj.get('reasoning_A', '')),
        'reasoning_B':        str(obj.get('reasoning_B', '')),
        'final_decision':     _normalize_decision(obj.get('final_decision', 'Tie')),
    }


class LLMJudgeEvaluator:

    def __init__(self,
                 language: str,
                 model_dir: Optional[str] = None,
                 repo: Optional[str] = None,
                 modelscope_id: Optional[str] = None,
                 tensor_parallel_size: Optional[int] = None,
                 dtype: Optional[str] = None,
                 quantization: Optional[str] = None,
                 enforce_eager: Optional[bool] = None,
                 max_model_len: Optional[int] = None,
                 gpu_memory_utilization: Optional[float] = None,
                 temperature: float = 0.0,
                 seed: int = 42,
                 max_tokens: int = 1024,
                 cache_path: Optional[str] = None):
        if language not in ('zh', 'en'):
            raise ValueError(f"language 必须是 'zh' (FUS) 或 'en' (CXR), 收到 {language!r}")
        self.language = language
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens

        self.model_dir = model_dir or getattr(config, 'JUDGE_LLM_DIR', None)
        self.repo      = repo or getattr(config, 'JUDGE_LLM_REPO', None)
        self.ms_id     = modelscope_id or getattr(config, 'JUDGE_LLM_MODELSCOPE', None)
        if not self.model_dir:
            raise ValueError('Judge 需要 config.JUDGE_LLM_DIR 或 model_dir 之一')

        def _cfg(name, default, override):
            return override if override is not None else getattr(config, name, default)
        self.tp                     = _cfg('JUDGE_LLM_TP',              4,             tensor_parallel_size)
        self.dtype                  = _cfg('JUDGE_LLM_DTYPE',           'float16',     dtype)
        self.quantization           = _cfg('JUDGE_LLM_QUANTIZATION',    'awq_marlin',  quantization)
        self.enforce_eager          = _cfg('JUDGE_LLM_ENFORCE_EAGER',   False,         enforce_eager)
        self.max_model_len          = _cfg('JUDGE_LLM_MAX_MODEL_LEN',   8192,          max_model_len)
        self.gpu_memory_utilization = _cfg('JUDGE_LLM_GPU_MEM_UTIL',    0.85,          gpu_memory_utilization)

        self._llm = None
        self._sampling = None

        self.cache_path = cache_path
        self._cache: Dict[str, Dict[str, Any]] = {}
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    self._cache = json.load(f)
                logging.info('Judge 缓存载入 %d 条 <- %s', len(self._cache), cache_path)
            except Exception as e:
                logging.warning('Judge 缓存读取失败 (%s), 忽略并重建: %s', cache_path, e)

    def _ensure_vllm(self) -> None:
        if self._llm is not None:
            return
        if _dir_has_weights(self.model_dir):
            logging.info('Judge 模型本地已就位, 跳过下载: %s', self.model_dir)
        else:
            _disable_hf_offline_for_download()
            os.makedirs(self.model_dir, exist_ok=True)
            ok = False
            if self.ms_id:
                ok = _modelscope_download(self.ms_id, self.model_dir)
            if not ok and self.repo:
                try:
                    from huggingface_hub import snapshot_download
                    logging.info('HF 下载 Judge 模型 %s -> %s', self.repo, self.model_dir)
                    snapshot_download(repo_id=self.repo, local_dir=self.model_dir,
                                      ignore_patterns=_hf_weight_ignore(self.repo))
                except Exception as e:
                    logging.warning('HF snapshot_download 失败: %s', e)
            if not _dir_has_weights(self.model_dir):
                raise FileNotFoundError(
                    f'{self.model_dir} 无 Judge 模型权重. 墙内请优先 ModelScope 预下: '
                    f'`modelscope download --model {self.ms_id or "<id>"} '
                    f'--local_dir {self.model_dir}`. Llama-3.3 在 HF 是 gated, 还需先在 hf.co '
                    f'接受 Meta 协议.')
        os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
        if self.enforce_eager:
            os.environ.setdefault('VLLM_USE_V1', '0')
            os.environ.setdefault('VLLM_ATTENTION_BACKEND', 'XFORMERS')
        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError(
                f'导入 vllm 失败: {type(e).__name__}: {e}  '
                f'(诊断: python -c "import vllm; print(vllm.__version__)")'
            ) from e
        logging.info('vLLM 加载 Judge 模型 (tp=%d, dtype=%s, quant=%s, eager=%s) ...',
                     self.tp, self.dtype, self.quantization, self.enforce_eager)
        self._llm = LLM(
            model=self.model_dir, tokenizer=self.model_dir,
            tensor_parallel_size=self.tp, dtype=self.dtype,
            quantization=self.quantization,
            trust_remote_code=True,
            enforce_eager=self.enforce_eager,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len, seed=self.seed,
        )
        self._sampling = SamplingParams(temperature=self.temperature,
                                         max_tokens=self.max_tokens, seed=self.seed)

    def _messages(self, findings: str, impression_A: str, impression_B: str):
        sys_p = JUDGE_SYSTEM_PROMPT_ZH if self.language == 'zh' else JUDGE_SYSTEM_PROMPT_EN
        usr_t = JUDGE_USER_TEMPLATE_ZH if self.language == 'zh' else JUDGE_USER_TEMPLATE_EN
        return [
            {'role': 'system', 'content': sys_p},
            {'role': 'user',   'content': usr_t.format(
                findings=findings, impression_A=impression_A, impression_B=impression_B)},
        ]

    def _cache_key(self, findings: str, A: str, B: str) -> str:
        return (f'{os.path.basename(self.model_dir)}\x00{self.language}\x00'
                f'{findings}\x00{A}\x00{B}')

    def _save_cache(self) -> None:
        if not self.cache_path:
            return
        try:
            os.makedirs(os.path.dirname(self.cache_path) or '.', exist_ok=True)
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f, ensure_ascii=False)
        except Exception as e:
            logging.warning('Judge 缓存落盘失败: %s', e)

    def _judge_batch_single_round(self,
                                   triplets: List[Tuple[str, str, str]]) -> List[Dict[str, Any]]:
        """对一批 (findings, A, B) 跑一轮 Judge: 命中缓存直接返回, 未命中走 vLLM 批量推理."""
        results: Dict[int, Dict[str, Any]] = {}
        todo_idx: List[int] = []
        todo_msgs: List[Any] = []
        for i, (f, a, b) in enumerate(triplets):
            k = self._cache_key(f, a, b)
            if k in self._cache:
                results[i] = self._cache[k]
            else:
                todo_idx.append(i)
                todo_msgs.append(self._messages(f, a, b))

        if todo_msgs:
            self._ensure_vllm()
            outputs = self._llm.chat(todo_msgs, self._sampling, use_tqdm=True)
            for i, o in zip(todo_idx, outputs):
                parsed = _parse_judge_json(o.outputs[0].text)
                f, a, b = triplets[i]
                self._cache[self._cache_key(f, a, b)] = parsed
                results[i] = parsed
            self._save_cache()

        return [results[i] for i in range(len(triplets))]

    @staticmethod
    def _combine_swap(orig_A: str, orig_B: str,
                      trial1: Dict[str, Any], trial2: Dict[str, Any]) -> Dict[str, Any]:

        d1 = trial1.get('final_decision')
        d2 = trial2.get('final_decision')
        if d1 == 'A' and d2 == 'B':
            v = 'A'
        elif d1 == 'B' and d2 == 'A':
            v = 'B'
        else:
            v = 'Tie'
        return {'verdict': v, 'trial1': trial1, 'trial2': trial2}

    def judge_pair(self, findings: str,
                   impression_A: str, impression_B: str) -> Dict[str, Any]:
        trials = self._judge_batch_single_round([
            (findings, impression_A, impression_B),
            (findings, impression_B, impression_A),
        ])
        return self._combine_swap(impression_A, impression_B, trials[0], trials[1])

    def evaluate(self,
                 findings_list: List[str],
                 predictions_A: List[str],
                 predictions_B: List[str]) -> Dict[str, Any]:

        assert len(findings_list) == len(predictions_A) == len(predictions_B), \
            (f'三列长度不一致: findings={len(findings_list)} '
             f'A={len(predictions_A)} B={len(predictions_B)}')
        n = len(findings_list)
        if n == 0:
            return {'win_rate_A': 0.0, 'win_rate_B': 0.0, 'tie_rate': 0.0,
                    'n_pairs': 0, 'per_pair': []}

        triplets = (
            [(f, a, b) for f, a, b in zip(findings_list, predictions_A, predictions_B)] +
            [(f, b, a) for f, a, b in zip(findings_list, predictions_A, predictions_B)]
        )
        all_results = self._judge_batch_single_round(triplets)

        wins_A = wins_B = ties = 0
        per_pair: List[Dict[str, Any]] = []
        for i in range(n):
            combined = self._combine_swap(
                predictions_A[i], predictions_B[i],
                all_results[i], all_results[i + n],
            )
            per_pair.append(combined)
            v = combined['verdict']
            if v == 'A':   wins_A += 1
            elif v == 'B': wins_B += 1
            else:          ties   += 1

        return {
            'win_rate_A': wins_A / n,
            'win_rate_B': wins_B / n,
            'tie_rate':   ties   / n,
            'n_pairs':    n,
            'per_pair':   per_pair,
        }


def compute_llm_judge_ab(findings_list: List[str],
                          predictions_A: List[str],
                          predictions_B: List[str],
                          language: str,
                          cache_path: Optional[str] = None,
                          **kwargs: Any) -> Dict[str, Any]:

    judge = LLMJudgeEvaluator(language=language, cache_path=cache_path, **kwargs)
    return judge.evaluate(findings_list, predictions_A, predictions_B)


def _print_rouge(name: str, res: Dict[str, float]) -> None:
    print(f'  样本数: {res["n_samples"]}')
    print(f'  {name} ROUGE-1  F1={res["rouge_1_f"]:.4f}  P={res["rouge_1_p"]:.4f}  R={res["rouge_1_r"]:.4f}')
    print(f'  {name} ROUGE-2  F1={res["rouge_2_f"]:.4f}  P={res["rouge_2_p"]:.4f}  R={res["rouge_2_r"]:.4f}')
    print(f'  {name} ROUGE-L  F1={res["rouge_l_f"]:.4f}  P={res["rouge_l_p"]:.4f}  R={res["rouge_l_r"]:.4f}')


if __name__ == '__main__':
    print('=' * 60)
    print('evaluate.py self-test —— ROUGE-1/2/L + LSC-F1 验证 (METEOR 已弃用)')
    print('=' * 60)

    print('\n[Test 1] 中文 ROUGE-1/2/L (FUS 数据集风格)')
    zh_preds = [
        '单胎, 头位, 胎儿大小相当于23周',
        '胎儿先天性心脏病, 室间隔缺损可能',
        '胎儿俯卧位, 各项参数未见明显异常',
    ]
    zh_refs = [
        '单胎头位, 胎儿大小相当于23W1D',
        '胎儿先天性心脏病: 室间隔缺损 (肌部)',
        '胎儿俯卧位, 各结构未见异常',
    ]
    res_zh = compute_rouge(zh_preds, zh_refs, language='zh')
    _print_rouge('zh', res_zh)
    assert 0 < res_zh['rouge_1_f'] < 1
    assert 0 < res_zh['rouge_2_f'] < 1
    assert 0 < res_zh['rouge_l_f'] < 1
    assert res_zh['rouge_1_f'] >= res_zh['rouge_2_f'], 'R-1 应不小于 R-2'

    print('\n[Test 2] 英文 ROUGE-1/2/L (MIMIC-CXR 数据集风格)')
    en_preds = [
        'No acute findings. Lungs are clear.',
        'Mild cardiomegaly with bilateral pleural effusions.',
        'Stable post-operative changes. No new lesions.',
    ]
    en_refs = [
        'No acute cardiopulmonary findings. Lungs clear.',
        'Cardiomegaly with bilateral pleural effusion.',
        'Postoperative changes are stable. No new lesion identified.',
    ]
    res_en = compute_rouge(en_preds, en_refs, language='en')
    _print_rouge('en', res_en)
    assert 0 < res_en['rouge_1_f'] < 1
    assert 0 < res_en['rouge_2_f'] < 1
    assert 0 < res_en['rouge_l_f'] < 1
    assert res_en['rouge_1_f'] >= res_en['rouge_2_f'], 'R-1 应不小于 R-2'

    print('\n[Test 3] 完美匹配 sanity check (中文)')
    perfect = compute_rouge(['完全一致的文本'], ['完全一致的文本'], language='zh')
    print(f'  R-1 F1={perfect["rouge_1_f"]:.4f}  R-2 F1={perfect["rouge_2_f"]:.4f}  R-L F1={perfect["rouge_l_f"]:.4f}')
    assert abs(perfect['rouge_1_f'] - 1.0) < 1e-6
    assert abs(perfect['rouge_2_f'] - 1.0) < 1e-6
    assert abs(perfect['rouge_l_f'] - 1.0) < 1e-6

    print('\n[Test 4] 向后兼容: compute_rouge_l 老接口')
    legacy = compute_rouge_l(zh_preds, zh_refs, language='zh')
    for key in ('rouge_l_p', 'rouge_l_r', 'rouge_l_f', 'n_samples'):
        assert key in legacy, f'缺 key: {key}'
    assert abs(legacy['rouge_l_f'] - res_zh['rouge_l_f']) < 1e-12
    print(f'  legacy rouge_l_f = {legacy["rouge_l_f"]:.4f} (与 compute_rouge 一致 ✓)')


    print('\n[Test 8] LSC-F1 纯数学核心 (substring, 复现论文 trace 数字)')
    gt_set = {
        _normalize_state(s) for s in
        ['右侧主支气管闭锁或梗阻可能', '心胸比例偏小', '左肺明显受压', '腹腔积液', '羊水过多']
    }
    gen_set = {_normalize_state(s) for s in ['左肺明显受压', '腹腔积液', '羊水正常']}
    prf = lsc_soft_match_prf(gen_set, gt_set, match_mode='substring')
    print(f'  TP={prf["tp"]}  P={prf["precision"]:.4f}  R={prf["recall"]:.4f}  F1={prf["f1"]:.4f}')
    assert prf['tp'] == 2
    assert abs(prf['precision'] - 2 / 3) < 1e-6
    assert abs(prf['recall'] - 0.4) < 1e-6
    assert abs(prf['f1'] - 0.5) < 1e-6

    print('\n[Test 9] LSC-F1 jieba_subset 模式 (复现论文 trace 的层级匹配)')
    gen_set2 = {_normalize_state(s) for s in ['左肺受压', '腹腔积液', '羊水正常']}
    prf_sub = lsc_soft_match_prf(gen_set2, gt_set, match_mode='substring')
    prf_jie = lsc_soft_match_prf(gen_set2, gt_set, match_mode='jieba_subset')
    print(f'  substring:   TP={prf_sub["tp"]} (左肺受压 匹配不到 -> TP=1)')
    print(f'  jieba_subset: TP={prf_jie["tp"]} F1={prf_jie["f1"]:.4f} (左肺受压⊆左肺明显受压 -> TP=2)')
    assert prf_sub['tp'] == 1
    assert prf_jie['tp'] == 2
    assert abs(prf_jie['f1'] - 0.5) < 1e-6

    print('\n[Test 10] LSC-F1 边界')
    both_empty = lsc_soft_match_prf(set(), set())
    assert both_empty['f1'] == 1.0
    gen_only = lsc_soft_match_prf({'羊水过多'}, set())
    assert gen_only['precision'] == 0.0 and gen_only['f1'] == 0.0
    print('  双空 F1=1.0 ✓; 过度诊断 (GT空) P=0 ✓')

    print('\n[Test 11] LLM-Judge 双盲互换规则')
    combos = [
        ('A', 'B', 'A',   'orig_A 两轮都赢 (trial1 投 A=orig_A, trial2 投 B=orig_A)'),
        ('B', 'A', 'B',   'orig_B 两轮都赢'),
        ('A', 'A', 'Tie', '位置偏好同侧 (都投 A, 矛盾 -> 平局)'),
        ('B', 'B', 'Tie', '位置偏好同侧 (都投 B)'),
        ('A', 'Tie', 'Tie', '一轮 Tie -> 不算赢'),
        ('Tie', 'B', 'Tie', '一轮 Tie -> 不算赢'),
        ('Tie', 'Tie', 'Tie', '都 Tie'),
    ]
    for d1, d2, expected, desc in combos:
        v = LLMJudgeEvaluator._combine_swap(
            'orig_A_text', 'orig_B_text',
            {'final_decision': d1}, {'final_decision': d2},
        )['verdict']
        assert v == expected, f'{desc}: 期望 {expected}, 得到 {v}'
        print(f'  trial1={d1!s:<4} trial2={d2!s:<4} -> {v!s:<4}  ✓ {desc}')

    print('\n' + '=' * 60)
    print('ROUGE-1/2/L + LSC-F1(数学核心) + Judge(双盲规则) 工作正常 ✓')
    print('=' * 60)
    res = compute_bertscore(
        ['单胎，头位，胎儿大小相当于23周'],
        ['单胎头位，胎儿大小相当于23W1D'],
    )
    print(f"BERTSCORE: {res}")
    res = compute_lsc_f1(
        ['单胎。左肺受压，腹腔积液。羊水正常。'],
        ['单胎。右侧主支气管闭锁或梗阻可能，心胸比例偏小，左肺明显受压，腹腔积液。羊水过多。'],
        match_mode='jieba_subset', cache_path='lsc_cache.json',
    )
    print(f"LSC-F1: {res}")
    
    res = compute_f1_chexbert(
        ['Cardiomegaly with bilateral pleural effusions.'],
        ['Enlarged cardiac silhouette and bilateral pleural effusion.'],
    )
    print('\n[F1-CheXbert] 14 类 micro-F1 = {:.4f}'.format(res['f1_chexbert']))
    res = compute_f1_radgraph(
        ['Cardiomegaly with bilateral pleural effusions.'],
        ['Enlarged cardiac silhouette and bilateral pleural effusion.'],
    )
    print(f"\n[F1-RadGraph] F1 = {res['f1_radgraph']:.4f} (reward_level='partial', model_type='radgraph')")
    res = compute_llm_judge_ab(
        findings_list=['单胎。AFV 86mm，BPD 91mm。心胸比例 0.39。羊水暗区 86mm。'],
        predictions_A=['模型A: 单胎妊娠，胎儿大小相当于34周，未见明显异常。'],
        predictions_B=['模型B: 单胎，孕37周，羊水过多，心胸比例偏小。'],
        language='zh',
        cache_path='judge_cache.json',
    )
    print(f"\n[Judge] A 胜率={res['win_rate_A']:.2f}  B 胜率={res['win_rate_B']:.2f}  Tie={res['tie_rate']:.2f}")
