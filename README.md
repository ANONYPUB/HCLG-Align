# HCLG-Align

Official implementation of **HCLG-Align: Neuro-Symbolic Alignment with
Heterogeneous Clinical Logic Graphs for Faithful Radiology Impression
Generation**.

HCLG-Align addresses findings-to-impression generation as a fusion problem
between free-text clinical findings and the structured medical logic contained
in them. It represents each findings section as a **Heterogeneous Clinical
Logic Graph (HCLG)**, encodes the graph with an R-GCN, and supplies the
projected node representations to a decoder-only language model through gated
cross-attention. During training, a contrastive objective with rule-corrupted
hard negative graphs encourages numerical, relational, and contextual
faithfulness.

## Main Contributions

- **Heterogeneous Clinical Logic Graph.** HCLG is a directed,
  multi-relational graph over clinical entities and observations. It contains
  structural attribution, co-diagnostic association, conditional dependency,
  and value-dependent discrepancy relations.
- **Neuro-symbolic graph--text alignment.** Magnitude-aware numerical node
  features, relational graph encoding, and gated node-level cross-attention
  make structured clinical information directly available during generation.
- **Clinically targeted contrastive learning.** Multi-level corruptions modify
  node attributes, graph relations, or contextual subgraphs to construct
  report-specific hard negatives.
- **Logical State Consistency (LSC-F1).** For MC-FUS, a fixed LLM extracts
  pathological-state sets from the generated and reference impressions. A
  deterministic one-to-one matching procedure then measures generated-only
  states through precision and omitted reference states through recall.

## Repository Structure

```text
.
├── data
│   ├── FUS
│   │   └── raw_data
│   │       ├── train_data.jsonl
│   │       ├── valid_data.jsonl
│   │       └── test_data.jsonl
│   └── CXR
│       └── raw_data
│           ├── train_data.jsonl
│           ├── valid_data.jsonl
│           └── test_data.jsonl
└── src_code
    └── model
        ├── build_HCLG.py
        ├── config.py
        ├── data.py
        ├── evaluate.py
        ├── hclg_verifier.py
        ├── model.py
        ├── predict.py
        └── train.py
```

## Runtime Environment

The implementation is designed for Linux systems with NVIDIA GPUs. Model
training in the paper used **four NVIDIA RTX 3090 Ti GPUs**. The 100-case
computational-cost measurements used four RTX 3090 Ti GPUs for Qwen3-14B
elicitation and one RTX 3090 Ti for graph encoding and generation. The commands
below illustrate a **two-GPU setup**; four-GPU settings are noted where relevant.
The principal dependencies are:

- Python 3.12
- PyTorch with a CUDA-compatible build
- PyTorch Geometric
- Hugging Face Transformers
- PEFT
- Accelerate
- vLLM and ModelScope for local HCLG elicitation and LSC-F1 extraction
- NumPy, tqdm, rouge-score, bert-score, jieba, and scikit-learn
- RadGraph and the CheXbert checkpoint for the complete MIMIC-CXR evaluation

Install versions compatible with the CUDA and PyTorch configuration of the
target machine. The local LLM tensor-parallel size, numerical precision, model
paths, and GPU-memory utilization are configured in
`src_code/model/config.py`.

## Data

### Included Synthetic Demonstration Data

Each train/validation/test file under `data/FUS/raw_data/` and
`data/CXR/raw_data/` contains **five fully synthetic examples** for inspecting
the expected JSONL schema and testing the pipeline. These records were written
from scratch to resemble the general structure and clinical reasoning patterns
of fetal-ultrasound and chest-radiography reports. They are not copied,
modified, or paraphrased from any MC-FUS or MIMIC-CXR record and contain no
patient, examination, institution, or date identifiers.

Real MC-FUS examples are not distributed because institutional privacy
regulations prohibit public release of the dataset. Real MIMIC-CXR reports are
also excluded because MIMIC-CXR is a credentialed-access resource whose license
does not permit redistribution to users who have not independently obtained
access. The included synthetic files are provided only as schema and pipeline
examples; they were not used in the paper experiments and cannot reproduce the
reported results. Their clinical content is fictitious and must not be used for
clinical decision-making.

Each record must contain at least:

```json
{"id": "synthetic_fus_train_001", "synthetic": true, "findings": "...", "impression": "..."}
```

### Full Datasets

- **MC-FUS** is a proprietary, multi-center Chinese fetal-ultrasound report
  dataset. The full dataset cannot be released because of institutional
  privacy restrictions.
- **MIMIC-CXR** is available through PhysioNet under its credentialed-data
  access requirements and license. To reproduce the supplementary experiment,
  prepare the official train/validation/test split as findings--impression
  JSONL records following the included CXR examples.

The paper uses 101,803/10,000/10,000 reports for the MC-FUS
train/validation/test partitions and the official 122,014/957/1,606
MIMIC-CXR split.

## Running HCLG-Align

All commands below are executed from `src_code/model/`:

```bash
cd src_code/model
```

### 1. Select the Dataset

Set the following option in `config.py`:

```python
DATASET = "FUS"  # "FUS" for MC-FUS or "CXR" for MIMIC-CXR
```

Dataset-specific paths, prompts, sequence limits, batch sizes, and evaluation
metrics are selected by this switch. Review the GPU-dependent settings in
`config.py` before running on a new machine.

### 2. Construct HCLGs

For the two-GPU example, run:

```bash
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 \
python build_HCLG.py
```

This command processes the selected dataset's train/validation/test files and
writes positive HCLGs and rule-corrupted negative graphs under
`data/<dataset>/raw_hclg/`. By default, schema-guided extraction uses a locally
deployed Qwen3-14B through in-process vLLM with
`HCLG_LLM_TP=2`. The model location, tensor-parallel size, precision, and the
optional OpenAI-compatible HTTP backend are configured in `config.py`.
For four-GPU elicitation, set `HCLG_LLM_TP=4` and
`CUDA_VISIBLE_DEVICES=0,1,2,3`.

The output is resumable: completed records are retained and failed records can
be retried by running the command again.

### 3. Train

For a quick run with the five synthetic examples per split, edit the selected
`FUS` or `CXR` branch in `src_code/model/config.py`: set `TRAIN_BATCH_SIZE = 1`,
`EVAL_BATCH_SIZE = 1`, `NUM_EPOCHS = 1`, and `EVAL_ROUGE_START_EPOCH = 1`; keep
`GRAD_ACCUM_STEPS = 1`. The small training batch prevents all five examples from
being dropped, and first-epoch validation enables saving the best checkpoint
used for prediction. Follow the graph-construction, training, and prediction
commands in this section; these settings are for checking pipeline execution,
not reproducing the paper's results.

Accelerate may be configured once for the local machine:

```bash
accelerate config
```

The two-GPU training example is:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
NCCL_DEBUG=INFO \
CUDA_VISIBLE_DEVICES=0,1 \
accelerate launch --num_processes=2 train.py
```

For the paper's four-GPU training setup, use
`CUDA_VISIBLE_DEVICES=0,1,2,3` and `--num_processes=4`.

Resume from the latest checkpoint with:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
NCCL_DEBUG=INFO \
CUDA_VISIBLE_DEVICES=0,1 \
accelerate launch --num_processes=2 train.py --resume
```

The best and latest checkpoints are stored under
`data/<dataset>/model/<model_name>/`.

### 4. Predict and Evaluate

The following example distributes prediction over two GPUs using the best
checkpoint and computes the dataset-specific metrics:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
NCCL_DEBUG=INFO \
CUDA_VISIBLE_DEVICES=0,1 \
accelerate launch --num_processes=2 predict.py
```

Useful alternatives are:

```bash
# Generate predictions without metric computation
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 \
accelerate launch --num_processes=2 predict.py --no-eval

# Recompute metrics from an existing test_predictions.jsonl.
# MC-FUS LSC-F1 uses both visible GPUs because LSC_LLM_TP=2.
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 \
python predict.py --eval-only

# MC-FUS: skip LSC-F1 when its local extractor is unavailable
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 \
accelerate launch --num_processes=2 predict.py --skip-lsc

# MIMIC-CXR: skip optional clinical metric dependencies
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 \
accelerate launch --num_processes=2 predict.py \
  --skip-chexbert --skip-radgraph
```

Predictions and metric summaries are saved under
`data/<dataset>/result/<model_name>/` as `test_predictions.jsonl` and
`eval_metrics.json`, respectively.

## Experimental Results

All values below are percentages and are copied from the corresponding tables
in the paper. They were obtained on the complete datasets, not on the five-case
demonstration files included in this repository.

### MC-FUS

| Category | Method | ROUGE-1 (%) | ROUGE-2 (%) | ROUGE-L (%) | BERTScore (%) | LSC-F1 (%) |
|:--|:--|--:|--:|--:|--:|--:|
| Graph-based | WGSUM (TRANS+GAT) | 77.19 | 67.79 | 76.86 | 87.27 | 57.36 |
| Graph-based | GECL | 81.08 | 72.55 | 80.86 | 88.24 | 57.19 |
| Pre-trained Enc-Dec | BART-large (fine-tuned) | 81.15 | 72.31 | 80.86 | 91.34 | 59.39 |
| Pre-trained Enc-Dec | T5-large (fine-tuned) | 81.05 | 70.36 | 80.89 | 92.30 | 67.38 |
| LLMs | LLaMA-3-8B (LoRA ft) | 88.30 | 81.63 | 88.00 | 94.07 | 74.11 |
| LLMs | Qwen3-8B (LoRA ft) | 88.03 | 81.37 | 87.74 | 93.93 | 73.60 |
| Ours | **HCLG-Align** | **89.49** | **83.31** | **89.21** | **94.66** | **76.88** |

### Supplementary MIMIC-CXR Evaluation

| Category | Method | ROUGE-1 (%) | ROUGE-2 (%) | ROUGE-L (%) | BERTScore (%) | F1-CheXbert (%) | F1-RadGraph (%) |
|:--|:--|--:|--:|--:|--:|--:|--:|
| Graph-based | WGSUM (TRANS+GAT) | 50.84 | 31.41 | 45.14 | 90.36 | 55.08 | 42.45 |
| Graph-based | GECL | 51.27 | 33.96 | 47.30 | 91.14 | 54.81 | 43.65 |
| Pre-trained Enc-Dec | BART-large (fine-tuned) | 49.96 | 33.97 | 47.74 | 91.54 | 59.81 | 43.38 |
| Pre-trained Enc-Dec | T5-large (fine-tuned) | 46.17 | 30.82 | 44.03 | 90.89 | 57.47 | 39.80 |
| LLMs | LLaMA-3-8B (LoRA ft) | 50.86 | 35.43 | 48.60 | 91.64 | 60.96 | 45.04 |
| LLMs | Qwen3-8B (LoRA ft) | 47.50 | 31.98 | 45.26 | 90.66 | 57.27 | 41.66 |
| Ours | **HCLG-Align** | **52.04** | **35.94** | **49.64** | **91.74** | **61.40** | **46.16** |

The MIMIC-CXR experiment evaluates the reusable graph-representation, fusion,
and contrastive components in qualitative chest radiology. It does not test the
fetal-specific formula-driven discrepancy mechanism.

## Configuration Notes

- The default generative backbone is `Qwen/Qwen3-0.6B`.
- The default HCLG elicitor is Qwen3-14B, deployed locally.
- Three hard negative graphs are sampled per positive graph during training.
- Dataset-specific settings, model paths, optimization hyperparameters,
  checkpoint directories, and generation options are centralized in
  `src_code/model/config.py`.

## Citation

Citation information will be added after publication.
