# DeepThinkLLM Example Scripts — README

This folder contains example scripts that demonstrate how to **run** DeepThinkLLM in different modes (online accelerated, baseline, offline) and how to **analyze** the resulting logs.

---

## Online Example

We provide a reproducible command for running DeepThinkLLM online:

```bash
python examples/example_online.py --qid $1 --rid $2 --dataset brumo_2025.jsonl --total_budget 256 --output_dir online-dpsk
```

This runs DeepThinkLLM in accelerated **online mode** with confidence-aware early-exit.  
The **voting algorithms** reported here are applied **after the online filtering step**, i.e., only traces that passed the online filter are considered for voting.

---

## Baseline Example

For a fair comparison, run the baseline (no early-exit):

```bash
python examples/example_online_baseline.py --qid $1 --rid $2 --dataset brumo_2025.jsonl --total_budget 256 --output_dir baseline-dpsk
```

### Baseline Results (Summary)

- **Accuracy:** ~90.0% (216/240)  
- **Token usage:** ~5.8M mean tokens  
- **Timing:** ~13.6k seconds mean generation time  

This shows the clear efficiency advantage of the online method.

---

## Online vs. Baseline (Evaluated on DeepSeek-8B with Brumo, 8 runs per problem, single A100 GPU)

| Mode      | Accuracy | Mean Tokens | Mean Gen Time |
|-----------|----------|-------------|---------------|
| **Online**   | **93.3%** (224/240) | ~2.7M | ~5.3k s |
| **Baseline** | 90.0% (216/240)     | ~5.8M | ~13.6k s |

---

## Analysis Utilities

### `example_analyze_online.py`
Analyze accelerated online runs. Example (already embedded in file):
```bash
python examples/example_analyze_online.py --output_dir ./online-dpsk/ --max_qid 29 --rids 1
```

### `example_analyze_online_baseline.py`
Analyze baseline runs. Example:
```bash
python examples/example_analyze_online_baseline.py --output_dir ./baseline-dpsk/ --max_qid 29 --rids 1
```

Both analyzers aggregate accuracy, tokens, and timing across `(qid, rid)` runs.

---

## Offline Script

### `example_offline.py`
Runs DeepThinkLLM in **offline mode** to generate full traces for a single question.  
Useful for debugging, ablation, or inspecting verifier behavior.

---

## Dataset Preparation

You can prepare datasets such as **AIME 2025** into JSONL format for these scripts:

```python
import json
from datasets import load_dataset

# Load dataset
dataset = load_dataset("MathArena/aime_2025", split="train")

# Convert to JSONL
with open("aime_2025.jsonl", "w", encoding="utf-8") as f:
    for example in dataset:
        entry = {
            "question": example["problem"],
            "answer": str(example["answer"])
        }
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Converted {len(dataset)} examples to aime_2025.jsonl")
```

---

## Requirements

- Python 3.10+
- [`vllm`](https://github.com/vllm-project/vllm)
- `deepconf` (DeepThinkLLM)
- `dynasor` (for `math_equal`)
- `pandas`, `numpy`, `tqdm`

Install extras with:
```bash
pip install pandas numpy tqdm
```

