from config import *
import config
import os, json, re, copy, random, time, traceback
from typing import List, Optional
from tqdm import tqdm



GA_TABLES = {
    "CRL": {5:6, 10:7.2, 15:8, 20:8.6, 25:9.2, 30:9.7, 40:10.7, 50:11.6, 60:12.4, 70:13.2, 80:14},
    "BPD": {20:12, 25:13, 30:15, 35:16, 40:17, 45:19, 50:20, 55:22, 60:23, 65:25, 70:27, 75:29, 80:31, 85:33, 88:35, 90:36, 92:37, 95:39},
    "HC":  {100:13, 130:15, 160:17, 190:19, 210:21, 240:23, 260:25, 280:27, 295:29, 310:31, 320:33, 330:35, 340:37},
    "AC":  {70:13, 100:15, 130:17, 160:19, 185:21, 210:23, 235:25, 255:27, 275:29, 290:31, 305:33, 320:35, 335:37},
    "FL":  {10:13, 15:15, 20:17, 25:19, 30:21, 35:23, 40:25, 45:27, 50:29, 55:31, 60:33, 65:35, 70:37, 75:39},
    "HL":  {10:13, 15:15, 20:17, 25:19, 30:21, 35:23, 40:25, 45:27, 50:29, 55:31, 60:33, 65:35, 70:37},
}
GA_DISCREPANCY_THRESHOLD = 2.0
MAX_PER_GROUP = 12


PROMPT_ZH = {
"system": """你是一位资深超声影像专家。请从【检查所见】中提取结构化信息。

输出一个JSON对象，包含两个字段：

## 1. "entities" 数组
每个元素代表一个独立的观测项：
- "entity": 结构/测量项名称
- "observation": 原文中的测量值或描述（保留原文表述）
- "negated_observation": 该 observation 在临床上的反义版本（用于负样本生成）
    * 对于定性描述：将关键状态/方位/存在性反转
      例: "未见明显异常" → "可见明显异常"
          "心律齐"       → "心律不齐"
          "形态规则，边界清晰" → "形态不规则，边界不清晰"
          "未见连续中断" → "见连续中断"
          "盖过内口"     → "未盖过内口"
    * 对于数值测量：保留原文不变（数值扰动由代码处理），即填与 observation 相同的内容
- "values": 数值列表。多维测量如"46×43×59mm"填[46,43,59]；单值如"85mm"填[85]；定性描述填null
- "unit": 单位字符串（定性填null）
- "group": 临床分组（见下方）
- "ga_type": 若该测量可用于估算孕周，填标准缩写（CRL/BPD/HC/AC/FL/HL）；否则填null

### 分组规则
从以下选择，对于左右两侧分别测量的结构，在组名后加_left或_right：
  fetal_biometry: 胎儿生物测量（径线、周长、骨长、胚囊、胚芽长等）
  amniotic_fluid: 羊水
  fetal_cardiac: 胎心
  placenta: 胎盘
  umbilical_cord: 脐带
  fetal_structure: 胎儿器官结构形态
  uterus: 子宫、宫腔
  uterine_artery: 子宫动脉（加_left/_right）
  adnexa: 附件、卵巢（加_left/_right）
  pelvic: 盆腔
  other: 以上均不适用

### 拆分规则
- 同一结构的多维尺寸（如"46×43×59mm"）保持为一个元素，values为列表
- 含义不同的独立参数（如PSV、S/D、PI、RI）必须拆为独立元素
- 同一结构的多个定性描述（如"形态规则，边界清晰"）可合并为一个元素

## 2. "dependencies" 数组
表示"一个观测的临床解读依赖于另一个"：
- "from": 被依赖方的entity名
- "to": 依赖方的entity名
- "reason": 简短原因

只输出JSON，不要Markdown或额外解释。

## 示例

输入：子宫：前位，外形：增大。宫腔：宫腔内见胚囊，大小46×43×59mm，胚芽长：25mm，心率：150次/分钟。附件：左卵巢：内见，27×27×26mm，形态规则，边界清晰，透声清晰，CDFI：囊壁见稀疏血流信号。右卵巢：内未见明显异常回声。盆腔：盆腔内未见游离无回声。

输出：
{
  "entities": [
    {"entity":"子宫","observation":"前位，外形增大","negated_observation":"后位，外形正常","values":null,"unit":null,"group":"uterus","ga_type":null},
    {"entity":"胚囊","observation":"46×43×59mm","negated_observation":"46×43×59mm","values":[46,43,59],"unit":"mm","group":"fetal_biometry","ga_type":null},
    {"entity":"胚芽长","observation":"25mm","negated_observation":"25mm","values":[25],"unit":"mm","group":"fetal_biometry","ga_type":"CRL"},
    {"entity":"胎心率","observation":"150次/分钟","negated_observation":"150次/分钟","values":[150],"unit":"次/分钟","group":"fetal_cardiac","ga_type":null},
    {"entity":"左卵巢囊肿","observation":"27×27×26mm，形态规则，边界清晰，透声清晰","negated_observation":"27×27×26mm，形态不规则，边界不清晰，透声差","values":[27,27,26],"unit":"mm","group":"adnexa_left","ga_type":null},
    {"entity":"左卵巢CDFI","observation":"囊壁见稀疏血流信号","negated_observation":"囊壁未见血流信号","values":null,"unit":null,"group":"adnexa_left","ga_type":null},
    {"entity":"右卵巢","observation":"未见明显异常回声","negated_observation":"可见明显异常回声","values":null,"unit":null,"group":"adnexa_right","ga_type":null},
    {"entity":"盆腔","observation":"未见游离无回声","negated_observation":"可见游离无回声","values":null,"unit":null,"group":"pelvic","ga_type":null}
  ],
  "dependencies": [
    {"from":"胚芽长","to":"胎心率","reason":"CRL>5mm时应可见胎心搏动"}
  ]
}""",
"user": "/no_think 请提取以下检查所见：\n{findings}",
}

PROMPT_EN = {
"system": """You are a senior radiologist. Extract structured information from the Findings text.

Output a JSON object with two fields:

## 1. "entities" array
Each element = one independent observation:
- "entity": structure or finding name
- "observation": original text of measurement or description
- "negated_observation": clinically negated version of the observation (used for negative sample generation)
    * For qualitative descriptions: invert key states/locations/presence
      e.g., "moderate, left, increased compared to prior" → "moderate, left, decreased compared to prior"
            "no pneumothorax" → "pneumothorax present"
            "heart size normal" → "heart size enlarged"
            "no consolidation" → "consolidation present"
    * For numerical measurements: keep identical to "observation" (numeric corruption handled by code)
- "values": list of numeric values. Multi-dimensional (e.g. "3.2 x 2.1 cm") → [3.2, 2.1]. Single value → [85]. Qualitative → null.
  IMPORTANT: MIMIC-CXR is de-identified, so numeric measurements often appear as the placeholder "___"
  (e.g. "___-mm nodular opacity", "___% loss of height", "several cm"). In ALL such cases set "values": null
  (NOT [___], NOT [several], NOT [None]). Only emit numeric literals when the original text contains an actual number.
- "unit": unit string (null if qualitative)
- "group": clinical category. For laterality-sensitive findings (pleural, pneumothorax,
  parenchymal, osseous), append _left or _right when the report specifies a side;
  use _bilateral when explicitly bilateral; omit suffix only when laterality is truly N/A.
  Categories: cardiac, pleural, parenchymal, pneumothorax, devices, osseous, other
- "ga_type": null (not applicable for chest X-rays)

### Splitting rules
- Multi-dimensional measurements of ONE structure → one element, values is a list
- Separate parameters with different meanings → separate elements
- A finding described separately for each side (e.g. "left pleural effusion ... right pleural effusion")
  → split into two elements with group=pleural_left and group=pleural_right

## 2. "dependencies" array
"Interpreting one finding depends on another":
- "from": the entity providing context
- "to": the entity whose interpretation depends on the other
- "reason": brief explanation

Output ONLY JSON, no markdown.

## Example

Input: Moderate left pleural effusion has increased compared to prior. Small right pleural effusion is stable. Associated compressive atelectasis at the left base. A ___-mm nodular opacity is seen in the right upper lobe. No pneumothorax. Heart size is normal.

Output:
{
  "entities": [
    {"entity":"left pleural effusion","observation":"moderate, increased compared to prior","negated_observation":"moderate, decreased compared to prior","values":null,"unit":null,"group":"pleural_left","ga_type":null},
    {"entity":"right pleural effusion","observation":"small, stable","negated_observation":"small, increased","values":null,"unit":null,"group":"pleural_right","ga_type":null},
    {"entity":"atelectasis","observation":"compressive, left base","negated_observation":"no atelectasis","values":null,"unit":null,"group":"parenchymal_left","ga_type":null},
    {"entity":"nodular opacity","observation":"___-mm, right upper lobe","negated_observation":"___-mm, right upper lobe","values":null,"unit":"mm","group":"parenchymal_right","ga_type":null},
    {"entity":"pneumothorax","observation":"none","negated_observation":"pneumothorax present","values":null,"unit":null,"group":"pneumothorax","ga_type":null},
    {"entity":"heart size","observation":"normal","negated_observation":"enlarged","values":null,"unit":null,"group":"cardiac","ga_type":null}
  ],
  "dependencies": [
    {"from":"left pleural effusion","to":"atelectasis","reason":"compressive atelectasis secondary to effusion"}
  ]
}""",
"user": "/no_think Extract from the following Findings:\n{findings}",
}


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL).strip()
    text = re.sub(r'^```json\s*', '', text)
    text = re.sub(r'^```\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m_start = text.find('{')
    m_end = text.rfind('}')
    if m_start >= 0 and m_end > m_start:
        try:
            return json.loads(text[m_start:m_end+1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Cannot parse JSON from LLM output: {text[:200]}")


def _prompt_for(dataset: str) -> dict:
    if dataset == "mc_fus":
        return PROMPT_ZH
    if dataset == "mimic_cxr":
        return PROMPT_EN
    raise ValueError(f"Unknown dataset: {dataset!r} (expect 'mc_fus' or 'mimic_cxr')")


def _messages(findings: str, prompt: dict) -> list:
    return [
        {"role": "system", "content": prompt["system"]},
        {"role": "user",   "content": prompt["user"].format(findings=findings)},
    ]


class _HCLGLLMExtractor:

    _instance: Optional["_HCLGLLMExtractor"] = None

    @classmethod
    def get(cls, backend: Optional[str] = None) -> "_HCLGLLMExtractor":
        """单例 (vllm 加载一次几分钟, 反复构造会重复加载). 切换 backend 时重建."""
        want = backend or getattr(config, 'HCLG_LLM_BACKEND', 'vllm')
        if cls._instance is None or cls._instance.backend != want:
            cls._instance = cls(want)
        return cls._instance

    def __init__(self, backend: str):
        if backend not in ('vllm', 'http'):
            raise ValueError(f"Unknown HCLG_LLM backend: {backend!r} (expect 'vllm' or 'http')")
        self.backend = backend
        if backend == 'vllm':
            self.model_dir       = config.HCLG_LLM_DIR
            self.repo            = getattr(config, 'HCLG_LLM_REPO', None)
            self.ms_id           = getattr(config, 'HCLG_LLM_MODELSCOPE', None)
            self.tp              = getattr(config, 'HCLG_LLM_TP', 1)
            self.dtype           = getattr(config, 'HCLG_LLM_DTYPE', 'bfloat16')
            self.enforce_eager   = getattr(config, 'HCLG_LLM_ENFORCE_EAGER', False)
            self.max_model_len   = getattr(config, 'HCLG_LLM_MAX_MODEL_LEN', 8192)
            self.gpu_mem_util    = getattr(config, 'HCLG_LLM_GPU_MEM_UTIL', 0.90)
            self._llm = None
            self._sampling = None
        else:
            from openai import OpenAI
            self.client = OpenAI(base_url=config.HCLG_LLM_URL,
                                 api_key=config.HCLG_LLM_KEY, timeout=600.0)
            self.model = config.HCLG_LLM_MODEL

    def _ensure_vllm(self) -> None:
        if self._llm is not None:
            return
        from evaluate import (_dir_has_weights, _modelscope_download,
                              _hf_weight_ignore, _disable_hf_offline_for_download)
        if _dir_has_weights(self.model_dir):
            print(f'[HCLG_LLM] 本地已就位: {self.model_dir}', flush=True)
        else:
            _disable_hf_offline_for_download()
            os.makedirs(self.model_dir, exist_ok=True)
            ok = False
            if self.ms_id:
                ok = _modelscope_download(self.ms_id, self.model_dir)
            if not ok and self.repo:
                try:
                    from huggingface_hub import snapshot_download
                    print(f'[HCLG_LLM] HF 下载 {self.repo} -> {self.model_dir}', flush=True)
                    snapshot_download(repo_id=self.repo, local_dir=self.model_dir,
                                      ignore_patterns=_hf_weight_ignore(self.repo))
                except Exception as e:
                    print(f'[HCLG_LLM] HF 下载失败: {e}', flush=True)
            if not _dir_has_weights(self.model_dir):
                raise FileNotFoundError(
                    f'{self.model_dir} 无可用权重. 墙内请先用 ModelScope 预下载:\n'
                    f'  modelscope download --model {self.ms_id or "<id>"} '
                    f'--local_dir {self.model_dir}')

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
        print(f'[HCLG_LLM] vLLM 加载 (tp={self.tp}, dtype={self.dtype}, eager={self.enforce_eager}) ...', flush=True)
        self._llm = LLM(
            model=self.model_dir, tokenizer=self.model_dir,
            tensor_parallel_size=self.tp, dtype=self.dtype,
            trust_remote_code=True,
            tokenizer_mode='slow',
            enforce_eager=self.enforce_eager,
            gpu_memory_utilization=self.gpu_mem_util,
            max_model_len=self.max_model_len, seed=42,
        )
        self._sampling = SamplingParams(temperature=0.8, max_tokens=8192, seed=1024)

    def _extract_one_http(self, findings: str, prompt: dict, max_retries: int = 3) -> Optional[dict]:
        last_err = None
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=_messages(findings, prompt),
                    temperature=0.01, max_tokens=8192,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                return _extract_json(resp.choices[0].message.content)
            except Exception as e:
                last_err = e
                if attempt < max_retries - 1:
                    time.sleep(2 ** (attempt + 1))
        print(f'[HCLG_LLM][http] 调用失败 (已重试 {max_retries} 次): {last_err}', flush=True)
        return None

    def extract_many(self, findings_list: List[str], dataset: str) -> List[Optional[dict]]:
        prompt = _prompt_for(dataset)
        if self.backend == 'vllm':
            self._ensure_vllm()
            convs = [_messages(f, prompt) for f in findings_list]
            outputs = self._llm.chat(convs, self._sampling, use_tqdm=True)
            results: List[Optional[dict]] = []
            for o in outputs:
                try:
                    results.append(_extract_json(o.outputs[0].text))
                except Exception as e:
                    print(f'[HCLG_LLM][vllm] JSON 解析失败: {e}', flush=True)
                    results.append(None)
            return results
        return [self._extract_one_http(f, prompt) for f in tqdm(findings_list, desc='[HCLG_LLM][http]')]

    def extract_one(self, findings: str, dataset: str) -> Optional[dict]:
        return self.extract_many([findings], dataset)[0]


def _interp_ga(value: float, table: dict) -> Optional[float]:
    pts = sorted(table.items())
    if value <= pts[0][0]:  return float(pts[0][1])
    if value >= pts[-1][0]: return float(pts[-1][1])
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]; x1, y1 = pts[i+1]
        if x0 <= value <= x1:
            return y0 + (value - x0) / (x1 - x0) * (y1 - y0)
    return None

def _build_graph(llm_out: dict) -> dict:
    ents = llm_out.get("entities", [])
    deps = llm_out.get("dependencies", [])
    nodes, edges = [], []
    name2oid, grp2oids = {}, {}

    for i, p in enumerate(ents):
        eid, oid = f"e{i}", f"o{i}"
        grp = p.get("group", "other")
        nodes.append({"id": eid, "type": "entity", "text": p["entity"], "group": grp})
        nodes.append({"id": oid, "type": "observation", "text": p["observation"],
                       "negated_text": p.get("negated_observation"),
                       "values": p.get("values"), "unit": p.get("unit"),
                       "group": grp, "ga_type": p.get("ga_type")})
        edges.append({"src": eid, "dst": oid, "rel": "has_obs"})
        name2oid[p["entity"]] = oid
        if grp and grp != "other":
            grp2oids.setdefault(grp, []).append(oid)

    for oids in grp2oids.values():
        if len(oids) > MAX_PER_GROUP:
            oids = random.sample(oids, MAX_PER_GROUP)
        for i in range(len(oids)):
            for j in range(i+1, len(oids)):
                edges.append({"src": oids[i], "dst": oids[j], "rel": "co_diagnostic"})
                edges.append({"src": oids[j], "dst": oids[i], "rel": "co_diagnostic"})

    for d in deps:
        s, t = name2oid.get(d.get("from")), name2oid.get(d.get("to"))
        if s and t:
            edges.append({"src": s, "dst": t, "rel": "conditional", "meta": d.get("reason","")})

    ga = {}
    for p in ents:
        gt = p.get("ga_type")
        vals = p.get("values")
        if gt and gt in GA_TABLES and vals:
            w = _interp_ga(vals[0], GA_TABLES[gt])
            if w is not None:
                ga[name2oid[p["entity"]]] = (p["entity"], w)
    ids = list(ga.keys())
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            oi, oj = ids[i], ids[j]
            ni, wi = ga[oi]; nj, wj = ga[oj]
            if abs(wi - wj) > GA_DISCREPANCY_THRESHOLD:
                m = f"{ni}→{wi:.1f}w vs {nj}→{wj:.1f}w"
                edges.append({"src":oi,"dst":oj,"rel":"discrepancy","meta":m})
                edges.append({"src":oj,"dst":oi,"rel":"discrepancy","meta":m})

    ec = {}
    for e in edges: ec[e["rel"]] = ec.get(e["rel"], 0) + 1
    return {"nodes": nodes, "edges": edges,
            "stats": {"num_nodes": len(nodes), "num_edges": len(edges),
                      "edge_type_counts": ec,
                      "has_discrepancy": any(e["rel"]=="discrepancy" for e in edges)}}


def _gen_negatives(pos: dict, k: int = 3) -> List[dict]:
    fns = [_corrupt_values, _corrupt_edges, _inject_subgraph]
    negs = []
    seen_sigs = set()
    seen_sigs.add(_graph_sig(pos))

    max_attempts = k * 5
    attempts = 0
    while len(negs) < k and attempts < max_attempts:
        attempts += 1
        neg = copy.deepcopy(pos)
        for fn in random.sample(fns, random.choice([1, 2])):
            neg = fn(neg)
        sig = _graph_sig(neg)
        if sig not in seen_sigs:
            seen_sigs.add(sig)
            negs.append(neg)

    while len(negs) < k:
        neg = copy.deepcopy(pos)
        neg = _inject_subgraph(neg)
        neg = _corrupt_values(neg)
        negs.append(neg)

    return negs


def _graph_sig(g: dict) -> str:
    node_texts = sorted(f"{n['id']}:{n.get('text','')}" for n in g["nodes"])
    edge_strs = sorted(f"{e['src']}-{e['rel']}-{e['dst']}" for e in g["edges"])
    return "|".join(node_texts) + "||" + "|".join(edge_strs)


def _corrupt_values(g):
    numeric = [n for n in g["nodes"] if n["type"]=="observation" and n.get("values")]
    negatable = [n for n in g["nodes"]
                  if n["type"]=="observation"
                  and not n.get("values")
                  and n.get("negated_text")
                  and n.get("negated_text") != n.get("text")]

    changed = False

    if numeric:
        for t in random.sample(numeric, min(2, len(numeric))):
            idx = random.randrange(len(t["values"]))
            v = t["values"][idx]
            offset = max(5, v * random.uniform(0.2, 0.5)) * random.choice([-1,1])
            t["values"][idx] = round(max(0.1, v + offset), 1)
            nums = re.findall(r'\d+\.?\d*', t["text"])
            if idx < len(nums):
                t["text"] = t["text"].replace(nums[idx], str(t["values"][idx]), 1)
            changed = True

    if not changed and negatable:
        t = random.choice(negatable)
        t["text"] = t["negated_text"]
        changed = True

    return g


def _corrupt_edges(g):
    modified = False

    co = [(i,e) for i,e in enumerate(g["edges"]) if e["rel"]=="co_diagnostic"
          and not e.get("meta")]
    if len(co) >= 2:
        idx, edge = random.choice(co)
        rev = {i for i,e in co if e["src"]==edge["dst"] and e["dst"]==edge["src"]}
        g["edges"] = [e for i,e in enumerate(g["edges"]) if i not in ({idx}|rev)]
        modified = True

    obs = [n for n in g["nodes"] if n["type"]=="observation"]
    by_g = {}
    for n in obs: by_g.setdefault(n.get("group","x"), []).append(n["id"])
    gs = list(by_g.keys())
    if len(gs) >= 2:
        g1, g2 = random.sample(gs, 2)
        n1, n2 = random.choice(by_g[g1]), random.choice(by_g[g2])
        existing = {(e["src"],e["dst"]) for e in g["edges"]}
        if (n1,n2) not in existing:
            g["edges"].append({"src":n1,"dst":n2,"rel":"co_diagnostic","meta":"corrupted"})
            g["edges"].append({"src":n2,"dst":n1,"rel":"co_diagnostic","meta":"corrupted"})
            modified = True

    if not modified and obs:
        n = random.choice(obs)
        g["edges"].append({"src":n["id"],"dst":n["id"],"rel":"co_diagnostic","meta":"corrupted"})

    return g


def _inject_subgraph(g):
    entities = [n for n in g["nodes"]
                if n["type"] == "entity" and n.get("text")
                and n.get("text") != "_injected_"]
    obs = [n for n in g["nodes"] if n["type"] == "observation"]
    if not entities or not obs:
        raise ValueError("L3 subgraph injection requires at least one entity-observation pair")

    mx = max(int(n["id"][1:]) for n in g["nodes"])
    eid, oid = f"e{mx+1}", f"o{mx+1}"

    entity_text = random.choice(entities)["text"]
    donor = random.choice(obs)
    donor_text = donor["text"]
    donor_values = copy.deepcopy(donor.get("values"))
    donor_unit = donor.get("unit")

    g["nodes"].append({"id":eid,"type":"entity","text":entity_text,"group":"hallucinated"})
    g["nodes"].append({"id":oid,"type":"observation","text":donor_text,
                        "values":donor_values,"unit":donor_unit,
                        "group":"hallucinated"})
    g["edges"].append({"src":eid,"dst":oid,"rel":"has_obs","meta":"hallucinated"})
    return g


def findings_to_hclg(findings: str, dataset: Optional[str] = None, k_neg: int = 3,
                      backend: Optional[str] = None) -> dict:
    if dataset is None:
        dataset = config.BUILD_HCLG_DATASET
    llm_out = _HCLGLLMExtractor.get(backend).extract_one(findings, dataset)
    if llm_out is None:
        raise RuntimeError("LLM 抽取失败 (JSON 解析错误或调用异常)")
    pos = _build_graph(llm_out)
    negs = _gen_negatives(pos, k_neg)
    return {"eo_pairs": llm_out, "positive_graph": pos, "negative_graphs": negs}


def raw_data_to_raw_hclg(raw_data_path: str, raw_hclg_path: str,
                          dataset: Optional[str] = None, k_neg: int = 3,
                          resume: bool = True,
                          backend: Optional[str] = None,
                          chunk_size: int = 200):
    if dataset is None:
        dataset = config.BUILD_HCLG_DATASET

    with open(raw_data_path, "r", encoding='utf-8') as f:
        all_data = [json.loads(line) for line in f if line.strip()]
    total = len(all_data)

    out_dir = os.path.dirname(raw_hclg_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    failed_path = os.path.splitext(raw_hclg_path)[0] + ".failed.jsonl"

    done_keys = set()
    if resume and os.path.exists(raw_hclg_path):
        with open(raw_hclg_path, "r", encoding='utf-8') as f:
            for line in f:
                try:
                    done_keys.add(json.loads(line)["id"])
                except Exception:
                    pass

    pending = [item for item in all_data
               if item.get("id") and item["id"] not in done_keys]
    print(f"[{os.path.basename(raw_data_path)}] Total={total}, Done={len(done_keys)}, "
          f"Pending={len(pending)}, ChunkSize={chunk_size}")
    if not pending:
        print("  Nothing to process.\n")
        return

    start = time.time()
    extractor = _HCLGLLMExtractor.get(backend)
    success, failed = 0, []
    n_chunks = (len(pending) + chunk_size - 1) // chunk_size

    with open(raw_hclg_path, "a", encoding='utf-8') as fout:
        for ci in range(n_chunks):
            chunk = pending[ci*chunk_size : (ci+1)*chunk_size]
            findings_list = [(it.get("findings") or it.get("Findings") or "") for it in chunk]
            print(f"  Chunk {ci+1}/{n_chunks} ({len(chunk)} samples) ...", flush=True)

            llm_outs = extractor.extract_many(findings_list, dataset)

            for item, llm_out in zip(chunk, llm_outs):
                idx = item.get("id", "")
                findings = item.get("findings") or item.get("Findings") or ""
                impression = item.get("impression") or item.get("Impression") or ""
                try:
                    if llm_out is None:
                        raise RuntimeError("LLM 抽取失败 (JSON 解析错误)")
                    pos = _build_graph(llm_out)
                    negs = _gen_negatives(pos, k_neg)
                    fout.write(json.dumps({
                        "id": idx,
                        "findings": findings, "impression": impression,
                        "positive_graph": pos, "negative_graphs": negs,
                    }, ensure_ascii=False) + "\n")
                    success += 1
                except Exception as e:
                    failed.append({
                        "id": idx, "findings": findings, "impression": impression,
                        "error": str(e)[:500],
                    })

            fout.flush()
            os.fsync(fout.fileno())
            print(f"    chunk {ci+1}/{n_chunks} written: cum_ok={success}, cum_fail={len(failed)}",
                  flush=True)

    if failed:
        with open(failed_path, "w", encoding='utf-8') as f:
            for r in failed:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    elif os.path.exists(failed_path):
        os.remove(failed_path)

    elapsed = (time.time() - start) / 60
    print(f"  Done. Success={success}, Failed={len(failed)}, Time={elapsed:.1f}min")
    if failed:
        print(f"  Failed -> {failed_path} (LLM JSON 解析错误, 下次跑会自动重试)\n")
    else:
        print()


if __name__ == "__main__":
    raw_data_to_raw_hclg(raw_data_path=train_raw_data_path, raw_hclg_path=train_raw_hclg_path, k_neg=3)
    raw_data_to_raw_hclg(raw_data_path=valid_raw_data_path, raw_hclg_path=valid_raw_hclg_path, k_neg=3)
    raw_data_to_raw_hclg(raw_data_path=test_raw_data_path,  raw_hclg_path=test_raw_hclg_path,  k_neg=3)








