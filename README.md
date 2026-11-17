# Supplementary experiments for ECE285 Homework 2

This branch contains the follow-up GAN experiments supporting the final
[HW2 report on main](https://github.com/rsilvabuarque/ECE285_HW2_RSB/blob/main/HW2_report.pdf).
It contains no final report PDF. The original baseline implementation and the
code that rebuilds the final report's figures and tables are on `main`.

The experiments ask whether model capacity, generator/discriminator balance,
regularization, augmentation, averaged weights and longer training improve the
original WikiArt generator. The selected model keeps the original DCGAN
architecture and uses real-image horizontal flips, EMA weights, and late
learning-rate decay. Not every tested intervention helped.

## Install and inspect the measured results

Run commands from this branch's repository root:

```bash
python -m pip install -r requirements.txt
```

Training and Inception evaluation require suitable compute resources; use an
allocation on an HPC system. The wrapper does not submit scheduler jobs. It runs
commands sequentially with the current Python interpreter. Offline plotting and
metadata-only planning do not use a GPU.

The measured results are in `results/improvement/`:

| File or directory | What it contains |
|---|---|
| `experiments.json` | All 19 screening configurations and their exact flags |
| `all_runs.csv`, `all_epoch_metrics.csv` | All 43 recorded trajectories and epoch histories |
| `configs/`, `histories/` | Individual measured configurations and histories |
| `screening_freeze.json`, `extension_freeze.json`, `selection.json` | The successive fixed selection decisions |
| `fixed_nine_specs.csv` | The nine fixed procedures ranked with added seeds 123 and 314 |
| `paired_decay_same_epoch.csv` | Constant versus decaying continuations at matched epochs |
| `final_evaluation/` | Common-test metrics for original, fresh control and improved model |
| `selected_models.json`, `checkpoints/` | Exact selected inference checkpoints, variants and hashes |
| `generation/` | Common-latent samples, all 64 images per model, and provenance |
| `validation_context/` | Unchanged original model and disjoint real-versus-real comparisons |
| `native128_demo/` | Separate native-resolution experiment, excluded from RGB64 ranking |
| `attempt_census.*`, `attempt_evidence/` | Recorded attempts, including informative failures |

A continuation history repeats its parent's earlier epochs. There are 43
trajectories, not 43 independent initializations or a sum of all their epoch
counts. Full training states for new runs are created by the workflow below;
the published inference checkpoints are not substitutes for Adam/RNG states.

Numerical protocols and measured software versions are recorded in `protocol.json`
and the metric JSONs. The environment used Torch 2.13.0+cu130, torchvision
0.28.0+cu130, NumPy 2.4.4, SciPy 1.18.0 and Ignite 0.5.5. Use a compatible CUDA
installation. Different versions/hardware can change training trajectories and
numerical scores. Replotting the retained data does not require retraining.

## Generate from the improved model

```bash
python -m hw2_gan.generate --checkpoint results/improvement/checkpoints/winner.pt --weights ema --expected-generator-state-sha256 f1875d34815d68764ac5d8143a6b5bb093caae618e19f40366b27d62d60e99a6 --output-dir output/new_samples --num-samples 64 --seed 45 --device auto
```

Use seed 43 for the report's fixed latent draws. For the original and fresh
control, use the checkpoint, raw/EMA variant and state hash in
`results/improvement/selected_models.json`. A shared latent vector does not imply
matching semantic content across independently trained generators.

The original supplementary four diagnostic figures can be replotted offline:

```bash
python -m hw2_gan.improvement_figures --results-dir results/improvement --output-dir output/supplement_figures
```

To rebuild the final report's three figures and all tables, use its main-branch
script and frozen inputs in a separate worktree:

```bash
git worktree add ../ECE285_HW2_main origin/main
cd ../ECE285_HW2_main
python scripts/rebuild_report.py
```

## Repeat the complete experiment sequence

Download version 1 of [steubk/wikiart](https://www.kaggle.com/datasets/steubk/wikiart)
to `data/wikiart.zip`. The raw archive and pixel/feature caches are not included.
The following dry run prints all commands and decisions without creating an
output directory, importing Torch, or running a model:

```bash
python -m campaign.reproduce all --root output/reproduction --archive data/wikiart.zip --dry-run
```

The default **dynamic** mode repeats the scientific selection procedure using
newly measured validation results. Its winning family can differ from the
published one. To replay the published family/epoch/weight choices while
measuring new outputs, add `--selection-mode recorded`. This explicitly reuses
historical choices and is not a new independent search.

Run the phases in order inside a suitable compute allocation:

```bash
python -m campaign.reproduce prepare --root output/reproduction --archive data/wikiart.zip
python -m campaign.reproduce screen --root output/reproduction
python -m campaign.reproduce select-screen --root output/reproduction
python -m campaign.reproduce confirm --root output/reproduction
python -m campaign.reproduce extend42 --root output/reproduction
python -m campaign.reproduce select-extension --root output/reproduction
python -m campaign.reproduce extend-confirm --root output/reproduction
python -m campaign.reproduce select-final --root output/reproduction
python -m campaign.reproduce evaluate --root output/reproduction
python -m campaign.reproduce native128 --root output/reproduction --archive data/wikiart.zip
python -m campaign.reproduce summarize --root output/reproduction
```

Use the same root and selection mode throughout. `all` executes this sequence;
`--skip-final-test` omits final evaluation, and `--skip-native128` omits the
separate resolution demonstration. `--cache-dir PATH` reuses a compatible RGB64
cache. CPU inference/evaluation is supported, but the complete campaign uses CUDA: the
supplied historical full100 parent contains CUDA random-generator state, which
cannot silently be replaced by CPU sampling state. The fresh RGB128 cache is
about 4 GB.

The phases implement these controls:

1. **Prepare:** RGB64 processing and the original content-hash split, a frozen
   reference/latent plan, validation feature caches and the disjoint-real control.
   No final-test features are extracted here. Both pooled validation sets contain
   4,000 byte-unique images; the new test excludes the historical 4,096 references
   by index and raw-content hash.
2. **Screen:** 18 fresh seed-42 configurations, each with a two-epoch pilot and
   continuation to 200, plus the original full epoch-100 state continued to 200.
   `campaign/experiments.json` is the source of every configuration flag. The
   historical continuation is observational and ineligible for fresh confirmation.
3. **Select screen and confirm:** choose each fresh family's epoch/ordinary-or-EMA
   specification using seed 42 at epochs 100/150/200. Freeze the three strongest
   noncontrol families plus mandatory unaugmented control, then train those four
   with seeds 123 and 314. Their checkpoint opportunities remain fixed.
4. **Extend and select extensions:** preserve every full epoch-200 parent. Extend
   all four families to 300 at constant rates; also extend the discovery leader
   with linear rate decay. Seed 42 fixes extension epoch/weight choices at
   250/300 before the added-seed extensions start. Decay twins resume the exact
   same full parent as constant controls, preserving Adam moments and RNG state.
5. **Select final:** rank four base, four constant-extension and one decay
   specification by mean validation pooled FID for seeds 123/314. Use the
   median-validation seed among 42/123/314 as each procedure's representative.
   Select the best control procedure independently of the overall winner.
6. **Evaluate:** freeze checkpoint identities before extracting final-test
   features, then score the original, fresh control and winner with common
   references and latent draws. Do not choose new settings from these test scores.
7. **Native128 and summarize:** process the raw archive directly at 128 pixels
   and train the separate width-32 demonstration; this is not upscaled RGB64.
   Export all fresh histories/configurations and final metrics to `ROOT/analysis/`.

The exact legacy continuation uses `checkpoints/original_epoch100_full.pt`
(76,186,061 bytes; SHA256
`807e4c5c60863603e9a354fea3554cf7f8a71bef03e1e66ffc314a5fecb40f8f`).
This includes both networks, optimizer and random states. The original epoch-95
inference checkpoint cannot replace it. The workflow snapshots new full200
parents before any fork; do not overwrite or delete them.

Completed phases can reuse matching frozen metadata and complete checkpoints.
Partial outputs are preserved rather than silently overwritten. A final-test
invocation creates a once-only evaluation claim; an interrupted evaluation needs
explicit inspection/recovery from retained feature artifacts, not deletion of
its claim. Run `python -m campaign.reproduce --help` for path overrides.

## Re-evaluate just the retained selected models

After `prepare` creates a compatible plan, the portable checkpoint-to-feature
entry point can evaluate the saved release models without retraining:

```bash
python -m campaign.evaluate_selected --selection results/improvement/selected_models.json --cache-dir output/reproduction/cache/rgb64 --plan output/reproduction/reference_plan/reference_plan.json --feature-cache-dir output/reproduction/features --output-dir output/released_model_test --device cuda
```

This is an explicit evaluation of already selected models; it is not a new
selection experiment. Use a fresh output directory. Historical absolute paths
inside retained JSONs are provenance; fresh workflow outputs bind their own
local paths and identities. Existing result files are never edited to look like
fresh measurements.

## Verification

`python -m unittest campaign.tests.test_portable` checks matrix-to-CLI mapping,
trajectory counts, paired-parent topology and fixed-specification ranking using
metadata only. The original model/metric/resume tests are also retained under
`campaign/tests/`; their numerical checks belong on an appropriate compute host.
The new wrapper was checked with dry runs and synthetic selection fixtures; this
report revision did not retrain the campaign. The measured results above come
from the completed original campaign, not those synthetic fixtures.
