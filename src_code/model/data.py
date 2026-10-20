import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from config import (
    ASYMMETRIC_RELS, EDGE_DROPOUT_P, EDGE_DROPOUT_TARGET_REL,
    EDGE_TYPE_TO_ID, IGNORE_INDEX, K_NEG, K_NEG_PER_STEP,
    MAX_FINDINGS_TOKENS, MAX_IMPRESSION_TOKENS, MAX_NODE_TEXT_TOKENS,
    NODE_TYPE_TO_ID, NUM_WORKERS, PIN_MEMORY,
    PROMPT_PREFIX, PROMPT_SUFFIX, USE_LOG_TRANSFORM,
    HF_CACHE_DIR,
)

import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch


def _worker_init_fn(worker_id: int) -> None:
    base_seed = torch.initial_seed() % (2 ** 32)
    random.seed(base_seed)
    np.random.seed(base_seed)



class HCLGData(Data):

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == 'value_seg_ids':
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key in ('flat_values', 'value_seg_ids',
                   'node_text_ids', 'node_text_mask',
                   'node_type', 'is_numeric'):
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)



def _log_transform(v: float) -> float:
    if not USE_LOG_TRANSFORM:
        return float(v)
    return float(np.sign(v) * np.log1p(abs(v)))


def _node_text_for_embed(node: Dict[str, Any]) -> Tuple[str, bool, List[float]]:
    if node['type'] != 'observation':
        return node['text'], False, []

    values = node.get('values') or []
    values = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    unit = node.get('unit')

    if values and unit:
        return unit, True, list(values)
    if values and not unit:
        return node['text'], True, list(values)
    return node['text'], False, []


_NODE_TOKEN_CACHE: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}


def _batch_tokenize_node_texts(
    texts: List[str], tokenizer,
) -> Tuple[torch.Tensor, torch.Tensor]:
    miss_texts: List[str] = []
    for t in texts:
        if t not in _NODE_TOKEN_CACHE:
            miss_texts.append(t)

    if miss_texts:
        enc = tokenizer(
            miss_texts,
            padding='max_length',
            truncation=True,
            max_length=MAX_NODE_TEXT_TOKENS,
            return_tensors='pt',
            add_special_tokens=False,
        )
        miss_ids = enc['input_ids']
        miss_mask = enc['attention_mask'].bool()
        for i, t in enumerate(miss_texts):
            _NODE_TOKEN_CACHE[t] = (miss_ids[i].clone(), miss_mask[i].clone())

    N = len(texts)
    T = MAX_NODE_TEXT_TOKENS
    all_ids = torch.empty((N, T), dtype=torch.long)
    all_mask = torch.empty((N, T), dtype=torch.bool)
    for i, t in enumerate(texts):
        ids, mask = _NODE_TOKEN_CACHE[t]
        all_ids[i] = ids
        all_mask[i] = mask
    return all_ids, all_mask


def _process_graph(graph_dict: Dict[str, Any], tokenizer) -> HCLGData:
    nodes = graph_dict['nodes']
    edges = graph_dict['edges']
    N = len(nodes)

    id2idx = {n['id']: i for i, n in enumerate(nodes)}

    node_type = torch.empty(N, dtype=torch.long)
    is_numeric = torch.zeros(N, dtype=torch.bool)
    text_for_embed: List[str] = []
    flat_values: List[float] = []
    value_seg_ids: List[int] = []

    for i, n in enumerate(nodes):
        node_type[i] = NODE_TYPE_TO_ID[n['type']]
        text, has_num, vals = _node_text_for_embed(n)
        text_for_embed.append(text)
        if has_num:
            is_numeric[i] = True
            for v in vals:
                flat_values.append(_log_transform(v))
                value_seg_ids.append(i)

    node_text_ids, node_text_mask = _batch_tokenize_node_texts(text_for_embed, tokenizer)

    src_list, dst_list, type_list = [], [], []
    for e in edges:
        rel = e['rel']
        if rel not in EDGE_TYPE_TO_ID:
            continue
        s = id2idx.get(e['src'])
        d = id2idx.get(e['dst'])
        if s is None or d is None:
            continue
        src_list.append(s); dst_list.append(d); type_list.append(EDGE_TYPE_TO_ID[rel])
        if rel in ASYMMETRIC_RELS:
            rev = rel + '_rev'
            src_list.append(d); dst_list.append(s); type_list.append(EDGE_TYPE_TO_ID[rev])

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_type = torch.tensor(type_list, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty(0, dtype=torch.long)

    flat_values_t = (torch.tensor(flat_values, dtype=torch.float32)
                     if flat_values else torch.empty(0, dtype=torch.float32))
    value_seg_t = (torch.tensor(value_seg_ids, dtype=torch.long)
                   if value_seg_ids else torch.empty(0, dtype=torch.long))

    return HCLGData(
        edge_index=edge_index,
        edge_type=edge_type,
        node_type=node_type,
        is_numeric=is_numeric,
        node_text_ids=node_text_ids,
        node_text_mask=node_text_mask,
        flat_values=flat_values_t,
        value_seg_ids=value_seg_t,
        num_nodes=N,
    )



class HCLGDataset(Dataset):

    def __init__(self,
                 jsonl_path: str,
                 tokenizer,
                 mode: str = 'train',
                 k_neg: int = K_NEG):
        assert mode in ('train', 'val', 'test'), f'unknown mode: {mode}'
        self.tokenizer = tokenizer
        self.mode = mode
        self.k_neg = k_neg if mode == 'train' else 0

        self.jsonl_path = jsonl_path
        self._offsets: List[int] = []
        self._lengths: List[int] = []
        with open(jsonl_path, 'rb') as f:
            offset = 0
            for line in f:
                if line.strip():
                    self._offsets.append(offset)
                    try:
                        rec = json.loads(line)
                        L = (len(rec.get('findings', ''))
                             + len(rec.get('impression', ''))
                             + len(rec.get('positive_graph', {}).get('nodes', [])))
                    except (json.JSONDecodeError, KeyError, TypeError):
                        L = 500
                    self._lengths.append(L)
                offset += len(line)
        self._file = None
        logging.info('Indexed %d samples from %s (mode=%s, len_range=[%d, %d])',
                     len(self._offsets), jsonl_path, mode,
                     min(self._lengths) if self._lengths else 0,
                     max(self._lengths) if self._lengths else 0)

        self.prompt_pre_ids = tokenizer(
            PROMPT_PREFIX, add_special_tokens=False, return_tensors='pt'
        ).input_ids[0]
        self.prompt_suf_ids = tokenizer(
            PROMPT_SUFFIX, add_special_tokens=False, return_tensors='pt'
        ).input_ids[0]

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        if self._file is None:
            self._file = open(self.jsonl_path, 'rb')
        self._file.seek(self._offsets[idx])
        rec = json.loads(self._file.readline())

        pos_g = _process_graph(rec['positive_graph'], self.tokenizer)

        neg_gs: List[HCLGData] = []
        if self.mode == 'train' and self.k_neg > 0:
            neg_pool = rec.get('negative_graphs', [])[:self.k_neg]
            if neg_pool:
                k_sample = min(K_NEG_PER_STEP, len(neg_pool))
                sampled = random.sample(neg_pool, k=k_sample)
                for g in sampled:
                    neg_gs.append(_process_graph(g, self.tokenizer))


        findings_text = rec.get('findings', '')
        findings_ids = self.tokenizer(
            findings_text,
            max_length=MAX_FINDINGS_TOKENS,
            truncation=True,
            add_special_tokens=False,
            return_tensors='pt',
        ).input_ids[0]

        impression_text = rec.get('impression', '')
        impression_ids: Optional[torch.Tensor] = None
        if self.mode in ('train', 'val'):
            eos_id = self.tokenizer.eos_token_id
            enc_ids = self.tokenizer(
                impression_text,
                max_length=MAX_IMPRESSION_TOKENS - 1,
                truncation=True,
                add_special_tokens=False,
                return_tensors='pt',
            ).input_ids[0]
            impression_ids = torch.cat([
                enc_ids,
                torch.tensor([eos_id], dtype=enc_ids.dtype),
            ])

        return {
            'pos_graph':       pos_g,
            'neg_graphs':      neg_gs,
            'findings_ids':    findings_ids,
            'impression_ids':  impression_ids,
            'findings_text':   findings_text,
            'impression_text': impression_text,
            'id':              rec.get('id', idx),
        }



def make_collate_fn(tokenizer,
                    mode: str = 'train',
                    edge_dropout_p: float = EDGE_DROPOUT_P,
                    prompt_pre_ids: Optional[torch.Tensor] = None,
                    prompt_suf_ids: Optional[torch.Tensor] = None):

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    target_rel_id = EDGE_TYPE_TO_ID[EDGE_DROPOUT_TARGET_REL]

    def _drop_co_diag(graph_batch: Batch) -> Batch:

        if mode != 'train' or edge_dropout_p <= 0:
            return graph_batch
        et = graph_batch.edge_type
        if et.numel() == 0:
            return graph_batch
        co_mask = (et == target_rel_id)
        n_co = int(co_mask.sum())
        if n_co == 0:
            return graph_batch
        keep_co = torch.rand(n_co) > edge_dropout_p
        full_keep = torch.ones(et.shape[0], dtype=torch.bool)
        full_keep[co_mask] = keep_co
        graph_batch.edge_index = graph_batch.edge_index[:, full_keep]
        graph_batch.edge_type = graph_batch.edge_type[full_keep]
        return graph_batch

    def _pad_seq(seqs: List[torch.Tensor], pad_value: int
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        max_len = max(s.shape[0] for s in seqs)
        out_ids = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
        out_mask = torch.zeros((len(seqs), max_len), dtype=torch.bool)
        for i, s in enumerate(seqs):
            out_ids[i, :s.shape[0]] = s
            out_mask[i, :s.shape[0]] = True
        return out_ids, out_mask

    def collate(batch_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        pos_batch = Batch.from_data_list([b['pos_graph'] for b in batch_list])
        pos_batch = _drop_co_diag(pos_batch)

        neg_batch: Optional[Batch] = None
        k_neg = 0
        if mode == 'train' and batch_list[0]['neg_graphs']:
            flat_negs = [g for b in batch_list for g in b['neg_graphs']]
            neg_batch = Batch.from_data_list(flat_negs)
            neg_batch = _drop_co_diag(neg_batch)
            k_neg = len(batch_list[0]['neg_graphs'])

        findings_ids, findings_mask = _pad_seq(
            [b['findings_ids'] for b in batch_list], pad_value=pad_id
        )

        impression_ids = impression_mask = None
        if mode in ('train', 'val') and batch_list[0]['impression_ids'] is not None:
            impression_ids, impression_mask = _pad_seq(
                [b['impression_ids'] for b in batch_list], pad_value=pad_id
            )

        return {
            'pos_graph':       pos_batch,
            'neg_graph':       neg_batch,
            'findings_ids':    findings_ids,
            'findings_mask':   findings_mask,
            'impression_ids':  impression_ids,
            'impression_mask': impression_mask,
            'prompt_pre_ids':  prompt_pre_ids,
            'prompt_suf_ids':  prompt_suf_ids,
            'pad_id':          pad_id,
            'ignore_index':    IGNORE_INDEX,
            'ids':             [b['id'] for b in batch_list],
            'findings_text':   [b['findings_text'] for b in batch_list],
            'impression_text': [b['impression_text'] for b in batch_list],
            'k_neg':           k_neg,
        }

    return collate



from torch.utils.data import Sampler


class SortishSampler(Sampler):

    def __init__(self, lengths: List[int], chunk_size: int = 200, seed: int = 0):

        super().__init__()
        self.lengths = list(lengths)
        self.chunk_size = chunk_size
        self.seed = seed
        self.epoch = 0
        self._n = len(self.lengths)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch * 1009)

        perm = np.arange(self._n)
        rng.shuffle(perm)

        chunks: List[np.ndarray] = []
        for start in range(0, self._n, self.chunk_size):
            chunk = perm[start:start + self.chunk_size]
            chunk_lens = np.array([self.lengths[i] for i in chunk])
            sorted_chunk = chunk[np.argsort(chunk_lens, kind='stable')]
            chunks.append(sorted_chunk)

        order = np.arange(len(chunks))
        rng.shuffle(order)

        for ci in order:
            for idx in chunks[ci]:
                yield int(idx)

    def __len__(self):
        return self._n



def build_dataloader(jsonl_path: str,
                     tokenizer,
                     mode: str,
                     batch_size: int,
                     num_workers: int = NUM_WORKERS,
                     pin_memory: bool = PIN_MEMORY,
                     edge_dropout_p: Optional[float] = None,
                     use_bucket_sampler: bool = True,
                     ) -> DataLoader:

    dataset = HCLGDataset(jsonl_path=jsonl_path, tokenizer=tokenizer, mode=mode)

    p_drop = edge_dropout_p if edge_dropout_p is not None else EDGE_DROPOUT_P

    collate = make_collate_fn(
        tokenizer=tokenizer,
        mode=mode,
        edge_dropout_p=p_drop,
        prompt_pre_ids=dataset.prompt_pre_ids,
        prompt_suf_ids=dataset.prompt_suf_ids,
    )

    if mode == 'train' and use_bucket_sampler:
        sampler = SortishSampler(
            lengths=dataset._lengths,
            chunk_size=max(200, batch_size * 10),
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate,
            drop_last=True,
            worker_init_fn=_worker_init_fn,
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(mode == 'train'),
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate,
            drop_last=(mode == 'train'),
            worker_init_fn=_worker_init_fn,
        )



if __name__ == '__main__':
    from transformers import AutoTokenizer
    from config import (BACKBONE_MODEL_NAME, TRUST_REMOTE_CODE,
                         test_raw_hclg_path, train_raw_hclg_path,
                         EVAL_BATCH_SIZE, TRAIN_BATCH_SIZE)

    tok = AutoTokenizer.from_pretrained(
        BACKBONE_MODEL_NAME,
        trust_remote_code=TRUST_REMOTE_CODE,
        cache_dir=HF_CACHE_DIR,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print('=== TEST mode (no negatives, no impression target) ===')
    test_loader = build_dataloader(
        test_raw_hclg_path, tok, mode='test',
        batch_size=EVAL_BATCH_SIZE, num_workers=0
    )
    b = next(iter(test_loader))
    print(' pos_graph        :', b['pos_graph'])
    print(' neg_graph        :', b['neg_graph'])
    print(' findings_ids     :', tuple(b['findings_ids'].shape))
    print(' impression_ids   :', b['impression_ids'])
    print(' k_neg            :', b['k_neg'])
    print(' ids[:3]          :', b['ids'][:3])

    print('\n=== TRAIN mode (with negatives + edge dropout) ===')
    train_loader = build_dataloader(
        train_raw_hclg_path, tok, mode='train',
        batch_size=min(TRAIN_BATCH_SIZE, 2), num_workers=0
    )
    b = next(iter(train_loader))
    print(' pos_graph        :', b['pos_graph'])
    print(' neg_graph        :', b['neg_graph'])
    print(' findings_ids     :', tuple(b['findings_ids'].shape))
    print(' impression_ids   :', tuple(b['impression_ids'].shape))
    print(' k_neg            :', b['k_neg'])
    print(' prompt_pre_ids   :', b['prompt_pre_ids'].tolist()[:8], '...')
