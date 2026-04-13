# Label Smoothing Study Report

## Methods
- Dataset: CIFAR-10
- Model: ResNet18 adapted for CIFAR-10 (3x3 stem, no maxpool)
- Experiments: baseline CE vs CE with label smoothing (epsilon=0.1)
- Seeds: 42, 0, 17
- Epochs: 20

## Quantitative Results
- Baseline final val loss: 0.2828 +/- 0.0029
- Label smoothing final val loss: 0.6982 +/- 0.0033
- Baseline final accuracy: 0.9061 +/- 0.0004
- Label smoothing final accuracy: 0.9139 +/- 0.0003

## Figures
- results/validation_loss.png
- results/train_loss.png
- results/tsne_baseline.png
- results/tsne_ls.png

## Limitations
- Single dataset and architecture.
- Runtime and convergence can vary by hardware.
- Results should be interpreted as a controlled educational study.
