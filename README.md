# ECE285 HW2: Painting GAN

Read [HW2_report.pdf](HW2_report.pdf). Run commands from this repository's root.

## Install

```bash
python -m pip install -r requirements.txt
```

The measured package versions are recorded in `results/final_evaluation/metrics.json`
and the training provenance. CUDA is used by default for training/evaluation;
pass `--device cpu` for CPU execution. Generation accepts `--device auto`.

## Generate paintings

```bash
python -m hw2_gan.generate --checkpoint results/selected/best.pt --output-dir output/new_samples --num-samples 64 --seed 45 --device auto
```

This saves individual PNGs, grids, latent vectors, and checkpoint metadata. Use a
new output directory. The provided `results/generation` contains the report samples.

## Regenerate report figures

```bash
python -m hw2_gan.report_figures --run-dir results/selected --samples-dir results/generation --comparison-run results/lowerlr64 --output-dir output/report_figures
```

The figure builder reads saved histories and images. To rebuild the PDF, copy the
new figure assets into `report/figures/`, then run `latexmk -pdf HW2_report.tex`
from `report/`. The official NeurIPS style is included.

## Prepare data and train

Download version 1 of [steubk/wikiart](https://www.kaggle.com/datasets/steubk/wikiart)
to `data/wikiart.zip`, then prepare a new cache:

```bash
python -m hw2_gan.data --archive data/wikiart.zip --output-dir data/rgb64 --workers 8 --seed 42
```

The cache needs about 1 GB of RAM when loaded, in addition to model/optimizer
memory. The supplied dataset manifest records exact source names, hashes, and
splits. Its inventory fingerprint hashes names/raw digests only. Use a new cache
and feature-cache directory whenever preprocessing or split settings change;
resume only with the original cache protocol.

Reproduce the baseline's process boundaries and total 100-epoch budget:

```bash
python -m hw2_gan.train --cache-dir data/rgb64 --output-dir output/baseline64 --epochs 2 --learning-rate 0.0002
python -m hw2_gan.train --cache-dir data/rgb64 --output-dir output/baseline64 --epochs 50 --resume output/baseline64/last.pt
python -m hw2_gan.train --cache-dir data/rgb64 --output-dir output/baseline64 --epochs 100 --resume output/baseline64/last.pt
```

For the comparison arm, use a new output directory and `--learning-rate 0.0001`
in the first command, followed by the same 50/100 resume boundaries. Other defaults
match the recorded configurations. `last.pt` restores optimizers and random states;
`best.pt` contains the validation-selected model. Completed raw runs retain both.
The released checkpoints are the selected baseline and comparison model weights.

## Evaluate the frozen model

```bash
python -m hw2_gan.evaluate --checkpoint results/selected/best.pt --cache-dir data/rgb64 --output-dir output/new_test_evaluation --split test --num-samples 4096 --seed 144 --device cuda --save-probabilities
```

The saved metric is **1000-class-probability-feature FID**, plus aggregate
Inception Score; its values differ from 2048-feature FID. Exact preprocessing,
weights, seeds, counts, reference indices, and software versions are in
`results/final_evaluation/metrics.json`. Model selection is recorded in
`results/selected/selection.json`; both 100-epoch histories are included.
