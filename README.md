# ECE285 Homework 2: generating WikiArt paintings

The final [HW2 report](HW2_report.pdf) describes the original GAN experiment,
the follow-up comparisons, and their final evaluation. This branch retains the
original training implementation and results. The
[`additional_analysis` branch](https://github.com/rsilvabuarque/ECE285_HW2_RSB/tree/additional_analysis)
contains the supplementary experiments, improved checkpoints, and the workflow
for repeating them. The final report PDF is published only on `main`.

Run commands from the repository root. Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Training and Inception evaluation use a CUDA GPU by default. On an HPC system,
run them inside an appropriate allocation. Rebuilding figures from saved results
does not use a GPU or load model weights.

## Rebuild every figure and numerical table in the final report

```bash
python scripts/rebuild_report.py
cd report
latexmk -pdf -interaction=nonstopmode -halt-on-error HW2_report.tex
cp HW2_report.pdf ../HW2_report.pdf
```

The first command reads the original histories in `results/` and the frozen
supplementary measurements and RGB pixels in `report/inputs/improvement/`.
It writes three figures, three LaTeX tables, and `report/reported_numbers.json`.
The input inventory records source hashes. No dataset download or training is
needed to reproduce these plots and tables. LaTeX with the included NeurIPS style
is needed only for the PDF. To keep the tracked files untouched, use
`--output-dir output/report` and inspect the outputs there.

The initial comparison used FID on 1,000 Inception **class probabilities** with
1,024 validation images and a 4,096-image historical test. The later comparisons
use 2,048 **pooled features** and 4,000 images. Their FID scales differ. The final
report compares all three frozen models under the same later protocol; it does
not compare the historical 0.059504 directly with 37.50.

## Generate images with the original selected model

```bash
python -m hw2_gan.generate --checkpoint results/selected/best.pt --output-dir output/original_samples --num-samples 64 --seed 45 --device auto
```

This saves PNGs, latent vectors, and checkpoint metadata. Use a fresh output
directory. The original report samples remain in `results/generation/`.
For the improved model and the final report's common-latent comparison, use the
supplementary branch's generation commands and selected checkpoints.

## Repeat the original data preparation and two learning-rate runs

Download version 1 of [steubk/wikiart](https://www.kaggle.com/datasets/steubk/wikiart)
to `data/wikiart.zip`. The archive and pixel cache are not included.

```bash
python -m hw2_gan.data --archive data/wikiart.zip --output-dir data/rgb64 --workers 8 --seed 42
```

Preparation converts images to RGB, resizes the shorter edge to 64, center-crops
64×64, and scales training pixels to [-1,1]. The content-hash split keeps
identical-byte copies in the same fold. The original dataset manifest and summary
in `results/dataset/` record the source inventory and split. The pixel cache uses
about 1 GB of RAM, in addition to model and optimizer memory. Use new cache paths
when changing preprocessing or splits.

The original baseline used these process boundaries and 100 total epochs:

```bash
python -m hw2_gan.train --cache-dir data/rgb64 --output-dir output/baseline64 --epochs 2 --learning-rate 0.0002
python -m hw2_gan.train --cache-dir data/rgb64 --output-dir output/baseline64 --epochs 50 --resume output/baseline64/last.pt
python -m hw2_gan.train --cache-dir data/rgb64 --output-dir output/baseline64 --epochs 100 --resume output/baseline64/last.pt
```

Repeat in `output/lowerlr64`, changing only the first command's learning rate
to `0.0001`. These boundaries matter because the original sampler's state also
depends on the data-loader lifecycle. All other defaults match the recorded
configurations. `last.pt` restores both networks, Adam states and random states;
`best.pt` stores the validation-selected inference model. Both measured histories
and selected checkpoints are included. Selection minimized fixed validation
probability-feature FID across the two arms, choosing baseline epoch 95.

To evaluate that checkpoint under the **historical** test protocol:

```bash
python -m hw2_gan.evaluate --checkpoint results/selected/best.pt --cache-dir data/rgb64 --output-dir output/historical_test --split test --num-samples 4096 --seed 144 --device cuda --save-probabilities
```

This does not run the later pooled-feature comparison. Its metric implementation,
software versions, reference indices and seeds are in
`results/final_evaluation/metrics.json`. The original eight implementation modules
in `hw2_gan/` retain their training and evaluation behavior. Repeating training on
a different software/hardware stack can change the learned weights; the retained
measurements and image pixels reproduce the report figures without relying on
bit-identical retraining.

## Supplementary experiments

Use a separate checkout so the implementations do not overwrite one another:

```bash
git worktree add ../ECE285_HW2_additional origin/additional_analysis
cd ../ECE285_HW2_additional
python -m campaign.reproduce --help
```

That branch documents the complete screen, added-seed confirmations, paired
constant/decaying continuations, final selection and evaluation, and the separate
native-128 demonstration. Its retained metrics and checkpoints are the evidence
behind the refinement sections of the final report.
