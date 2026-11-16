# ECE285 HW2: Controlled WikiArt GAN experiments

Read [HW2_report.pdf](HW2_report.pdf). Commands run from this directory.
The matching development branch is [improve/gan-20261115](https://github.com/rsilvabuarque/ECE285_HW2_RSB/tree/improve/gan-20261115).

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Generate from the exact selected generator (use a fresh output directory):

```bash
python -m hw2_gan.generate --checkpoint results/improvement/checkpoints/winner.pt --weights ema --expected-generator-state-sha256 f1875d34815d68764ac5d8143a6b5bb093caae618e19f40366b27d62d60e99a6 --output-dir output/new_samples --num-samples 64 --seed 45 --device auto
```

Use `--seed 43 --num-samples 64` for the report's fixed latent vectors; select each
model's explicit variant and hash from `selected_models.json`. Seed45 gives new
draws. The figure command below replots the retained pixels.

Regenerate the report figures from retained pixels and measured histories:

```bash
python -m hw2_gan.improvement_figures --results-dir results/improvement --output-dir output/figures
```

Rebuild the report with `latexmk -pdf HW2_report.tex` in `report/`.
`results/improvement/selected_models.json` identifies original, matched-control
and winning weights, variants and hashes. `all_runs.csv`, `all_epoch_metrics.csv`,
`fixed_nine_specs.csv`, the selection manifests and final metric JSONs retain
the measured comparisons. `native128_demo/` is a separate native-resolution model.

Prepare fresh data and validation caches before training. The raw WikiArt v1
archive and pixel caches are not distributed:

```bash
python -m hw2_gan.data --archive data/wikiart.zip --output-dir data/rgb64 --workers 8 --seed 42
python -m hw2_gan.evaluate_protocols freeze-plan --cache-dir data/rgb64 --output-dir data/plan --historical-test-metrics results/improvement/historical_test4096_metrics.json --pixel-cache-sha256 YOUR_MEASURED_IMAGES_NPY_SHA256
python -m hw2_gan.evaluate_protocols precompute --cache-dir data/rgb64 --plan data/plan/reference_plan.json --feature-cache-dir data/features --references validation_legacy1024 validation_pool4000_a
python -m hw2_gan.campaign_train --cache-dir data/rgb64 --output-dir output/fresh_control --data-order epoch-v1 --ema-decay .999 --seed 42 --epochs 200 --evaluation-plan data/plan/reference_plan.json --feature-cache data/features
```

Per-run configuration JSONs and the experiment matrix record the flags for all
variants. Reproduce those flags and checkpoint opportunities when repeating an
arm. Fresh cache paths/metadata produce a new plan identity; verify the original
pixel checksum, raw hashes, split indices and prior vectors before comparing.
Full `last.pt` files contain training state; numbered checkpoint files are for
inference. For paired 200-to-300 training, preserve full200 immutably and resume
into an empty output directory; add `--fork-linear-decay` only for its declared
paired decay clone. Never select new settings from final test scores.

