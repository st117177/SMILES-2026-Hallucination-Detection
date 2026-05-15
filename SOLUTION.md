# SMILES-2026 Hallucination Detection Solution

## Reproducibility

Run the solution from a GPU runtime, for example Google Colab with a T4 GPU:

```bash
git clone https://github.com/st117177/SMILES-2026-Hallucination-Detection.git
cd SMILES-2026-Hallucination-Detection
pip install -r requirements.txt
python solution.py
```

The official `solution.py` script generates:

- `results.json`
- `predictions.csv`

The model used by the fixed infrastructure is `Qwen/Qwen2.5-0.5B`.

## Baseline

- Aggregation: final transformer layer, last real token.
- Probe: MLP with one hidden layer, `input_dim -> 256 -> 1`.
- Splitting: one stratified train/validation/test split.
- Feature dimension: 896.
- Total labelled samples: 689.
- Extract time: 161.9 seconds.

Metrics:

| Split | Accuracy | F1 | AUROC |
|---|---:|---:|---:|
| Majority baseline | 70.19% | 82.49% | N/A |
| Train | 86.07% | 90.96% | 100.00% |
| Validation | 72.12% | 82.63% | 66.81% |
| Internal test | 74.04% | 83.02% | 73.40% |

Observation: the baseline probe overfits. Train AUROC is 100.00%, while validation and internal test AUROC are much lower.

## Experiments

### Linear probe

Reason: reduce overfitting by replacing the hidden-layer MLP with a simpler linear classifier.

Change:

```python
self._net = nn.Sequential(
    nn.Linear(input_dim, 1),
)
```

Metrics:

| Split | Accuracy | F1 | AUROC |
|---|---:|---:|---:|
| Majority baseline | 70.19% | 82.49% | N/A |
| Train | 76.51% | 85.31% | 87.88% |
| Validation | 71.15% | 82.56% | 64.96% |
| Internal test | 69.23% | 80.95% | 67.61% |

Decision: discarded. The linear probe reduced overfitting, but it was too simple and lowered internal test accuracy from 74.04% to 69.23%.

### Regularized MLP

Reason: keep a nonlinear classifier, but reduce overfitting with lower capacity, dropout, and weight decay.

Change:

```python
self._net = nn.Sequential(
    nn.Linear(input_dim, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 1),
)
```

Training uses `AdamW` with `weight_decay=1e-2`.

Result: pending.
