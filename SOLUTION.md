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

Metrics:

| Split | Accuracy | F1 | AUROC |
|---|---:|---:|---:|
| Majority baseline | 70.19% | 82.49% | N/A |
| Train | 86.69% | 91.33% | 99.50% |
| Validation | 73.08% | 83.13% | 66.81% |
| Internal test | 74.04% | 83.02% | 74.11% |

Decision: kept for the next experiment. Internal test accuracy matched the baseline, validation accuracy improved from 72.12% to 73.08%, and internal test AUROC improved from 73.40% to 74.11%.

### Last-token plus mean pooling aggregation

Reason: the baseline uses only the final token from the final transformer layer. Hallucination evidence may be distributed across the whole prompt/response sequence, so the next experiment concatenates the final-token representation with mean-pooled token representations from the same layer.

Change:

```python
feature = torch.cat([last_feature, mean_feature], dim=0)
```

Metrics:

| Split | Accuracy | F1 | AUROC |
|---|---:|---:|---:|
| Majority baseline | 70.19% | 82.49% | N/A |
| Train | 80.25% | 87.65% | 100.00% |
| Validation | 71.15% | 82.95% | 66.55% |
| Internal test | 66.35% | 79.77% | 50.84% |

Decision: discarded. Concatenating last-token and mean-pooled features increased the feature dimension from 896 to 1792, but internal test accuracy dropped from 74.04% to 66.35% and internal test AUROC dropped from 74.11% to 50.84%.

## Current Selected Approach

- Aggregation: final transformer layer, last real token.
- Probe: regularized MLP with one 128-unit hidden layer, dropout, and AdamW weight decay.
- Splitting: one stratified train/validation/test split.
