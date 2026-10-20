import json
import math
from typing import Dict, List, Set, Tuple, Optional


def _is_chinese(text: str) -> bool:

    if not text:
        return False
    cjk = sum(1 for c in text if '一' <= c <= '鿿')
    return cjk / max(1, len(text)) > 0.3


def normalize_entity_text(text: str) -> str:

    if not text:
        return ''
    t = text.strip().lower()
    t = ' '.join(t.split())
    return t



def extract_hclg_entities(sample_or_graph) -> List[str]:

    if 'nodes' in sample_or_graph:
        nodes = sample_or_graph['nodes']
    elif 'positive_graph' in sample_or_graph:
        nodes = sample_or_graph['positive_graph'].get('nodes', [])
    else:
        return []

    entities: List[str] = []
    seen: Set[str] = set()
    for node in nodes:
        text = node.get('text', '')
        if not text:
            continue
        norm = normalize_entity_text(text)
        if norm and norm not in seen:
            seen.add(norm)
            entities.append(norm)
    return entities


def load_test_hclg_entities(jsonl_path: str) -> Dict[str, List[str]]:

    out: Dict[str, List[str]] = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            pg = sample.get('positive_graph')
            if not isinstance(pg, dict) or 'nodes' not in pg:
                continue
            sid = sample.get('id', '')
            if not sid:
                raise ValueError(
                    f'load_test_hclg_entities: sample at {jsonl_path} line {line_no} '
                    f'has missing or empty "id" field. TTC verifier 需要 id 做 lookup, '
                    f'请确认 data builder 给每个样本写了 id.'
                )
            out[sid] = extract_hclg_entities(pg)
    return out



class HCLGVerifier:

    def __init__(self,
                 target_length: float = 30.0,
                 length_sigma: float = 0.5,
                 length_weight: float = 0.5):

        self.target_length = float(target_length)
        self.length_sigma = float(length_sigma)
        self.length_weight = float(length_weight)

    def set_target_length(self, length: float) -> None:
        self.target_length = float(length)


    def score(self, candidate_text: str, hclg_entities: List[str]) -> float:

        if not hclg_entities:
            return 0.0
        if not candidate_text:
            return 0.0

        cov = self._entity_coverage(candidate_text, hclg_entities)
        if self.length_weight > 0:
            lmatch = self._length_match_score(candidate_text)
            return cov * (1.0 - self.length_weight + self.length_weight * lmatch)
        return cov

    def score_batch(self,
                     candidates: List[str],
                     hclg_entities: List[str]) -> List[float]:
        return [self.score(c, hclg_entities) for c in candidates]


    def _entity_coverage(self,
                          candidate_text: str,
                          hclg_entities: List[str]) -> float:

        cand_norm = normalize_entity_text(candidate_text)
        if not cand_norm:
            return 0.0

        hit = 0
        for ent in hclg_entities:
            if ent and ent in cand_norm:
                hit += 1
        return hit / max(1, len(hclg_entities))


    def _length_match_score(self, candidate_text: str) -> float:
        if _is_chinese(candidate_text):
            length = len(candidate_text) / 1.7
        else:
            length = len(candidate_text.split())

        if self.target_length <= 0:
            return 1.0
        ratio = length / self.target_length
        return math.exp(-((ratio - 1.0) ** 2) / max(1e-6, 2.0 * self.length_sigma ** 2))



class MBRScorer:

    def score_batch(self, candidates: List[str]) -> List[float]:
        K = len(candidates)
        if K <= 1:
            return [0.0] * K
        token_lists = [self._tokenize(c) for c in candidates]
        scores = []
        for i in range(K):
            total = 0.0
            count = 0
            for j in range(K):
                if i == j:
                    continue
                total += self._rouge_l(token_lists[i], token_lists[j])
                count += 1
            scores.append(total / max(1, count))
        return scores

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = (text or '').strip().lower()
        if not text:
            return []
        if _is_chinese(text):
            return [c for c in text if c.strip()]
        return text.split()

    @staticmethod
    def _rouge_l(tokens_a: List[str], tokens_b: List[str]) -> float:
        if not tokens_a or not tokens_b:
            return 0.0
        lcs_len = MBRScorer._lcs_length(tokens_a, tokens_b)
        if lcs_len == 0:
            return 0.0
        p = lcs_len / len(tokens_a)
        r = lcs_len / len(tokens_b)
        return 2 * p * r / max(1e-8, p + r)

    @staticmethod
    def _lcs_length(a: List[str], b: List[str]) -> int:
        n, m = len(a), len(b)
        if n == 0 or m == 0:
            return 0
        prev = [0] * (m + 1)
        for i in range(1, n + 1):
            curr = [0] * (m + 1)
            ai = a[i - 1]
            for j in range(1, m + 1):
                if ai == b[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(prev[j], curr[j - 1])
            prev = curr
        return prev[m]



def compute_target_length_from_jsonl(jsonl_path: str,
                                       max_samples: int = 5000) -> float:

    lengths: List[int] = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            impr = sample.get('impression', '')
            if impr:
                toks = impr.split()
                if len(toks) <= 2 and len(impr) > 10:
                    lengths.append(int(len(impr) / 1.7))
                else:
                    lengths.append(len(toks))
    if not lengths:
        return 30.0
    lengths.sort()
    return float(lengths[len(lengths) // 2])
