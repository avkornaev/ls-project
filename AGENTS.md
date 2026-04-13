# AGENTS.md — Codex

## Mission

Implement and run a controlled experiment comparing:

Baseline:

CrossEntropy

Variant:

Label Smoothing (ε = 0.1)

The goal is reproducible training and logging.

---

## Allowed Changes

You may modify:

train.py  
loss.py  
model.py  
analyze_results.py  

You may add:

plotting functions  
t-SNE computation  

---

## Forbidden Changes

Do not change:

dataset  
train/test split  
augmentation pipeline  
number of epochs  
learning rate schedule  
seed list  

unless explicitly instructed.

---

## Experiments

Run exactly:

3 seeds per experiment

Seeds:

42  
0  
17

---

## Training Rules

Use:

ResNet18 for CIFAR-10

Simple augmentations only:

RandomCrop  
RandomHorizontalFlip  

---

## Logging Requirements

Every epoch must log:

train_loss  
val_loss  
accuracy  
epoch  
seed  

---

## Visualization Requirements

Generate:

validation_loss.png

Plot:

epoch vs validation loss

Average across seeds.

---

## t-SNE Rules

Compute t-SNE once per experiment.

Use:

best validation checkpoint

Sample:

1000 test images

Output:

tsne_baseline.png  
tsne_ls.png  

---

## Output Directory

results/

---

## Execution Order

Step 1

Train baseline.

Step 2

Train label smoothing model.

Step 3

Aggregate results.

Step 4

Generate plots.

Step 5

Save metrics.

---

## Performance Target

Each experiment must complete in:

less than 10 minutes

on a single GPU.

---

## Reproducibility

All runs must:

set random seed  
log seed  
log config