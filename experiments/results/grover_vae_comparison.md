# Grover VAE vs Contrastive Trajectory Encoder Comparison

Seed: 42 | 500 pretrain agents | 100 eval agents | 200 epochs

## Kuhn Poker

| Metric | Contrastive | Grover VAE |
|--------|------------|------------|
| Task A MSE | 0.0426 | 0.1085 |
| Task A Baseline MSE | 0.1233 | 0.1013 |
| Task A Improvement | 65.5% | -7.1% |
| Task D k-NN Top-1 | 0.724 | 0.042 |
| Task D Linear Top-1 | 0.756 | 0.040 |

## Leduc Poker

| Metric | Contrastive | Grover VAE |
|--------|------------|------------|
| Task A MSE | 0.4707 | 0.4591 |
| Task A Baseline MSE | 0.5620 | 0.3731 |
| Task A Improvement | 16.2% | -23.0% |
| Task D k-NN Top-1 | 0.714 | 0.042 |
| Task D Linear Top-1 | 0.794 | 0.038 |

## Notes
- Random baseline for Task D: 1% (1/100 agents)
- Grover VAE trained with val_loss=0.9663 (Kuhn), best epoch 194
- Contrastive loaded from existing checkpoints
- VAE performs worse than mean baseline on Task A for both games
- VAE Task D accuracy barely above random
