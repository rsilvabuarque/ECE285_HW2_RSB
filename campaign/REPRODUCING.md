# Reproduce the supplementary experiments

Run commands from the repository root on the `additional_analysis` branch.
The wrapper uses the current Python interpreter and starts ordinary sequential
processes. It does not request an allocation or submit scheduler jobs. Numerical
phases belong on a compute host with sufficient memory and, normally, a CUDA GPU.

There are two different reproducibility tasks:

* Replot the measured report evidence without training:
  `python -m hw2_gan.improvement_figures --results-dir results/improvement --output-dir output/measured_figures`.
* Rerun the training and selection procedure below. New training is a new
  measurement; matching recorded seeds does not establish bitwise identity across
  different hardware, library versions or interruption boundaries.

The raw WikiArt v1 archive is not distributed. Put it at `data/wikiart.zip`, or
pass its location with `--archive`. The supplied full epoch100 checkpoint in
`checkpoints/original_epoch100_full.pt` is needed for the legacy continuation.
The three selected inference checkpoints in `results/improvement/checkpoints/`
support generation and evaluation, but cannot replace full optimizer/RNG states.

## Inspect the plan before executing it

These commands perform no training, model inference or feature extraction:

```bash
python -m campaign.tests.test_portable
python -m campaign.reproduce all --root output/reproduction --archive data/wikiart.zip --dry-run
```

The second command prints all process arguments and parent-copy operations.
Before fresh scores exist, its dynamic-mode dry run uses the recorded shortlist
to illustrate the later topology, explicitly marked as a planning assumption.
An actual dynamic run chooses the shortlist from its own new validation scores.

## Run the full procedure

This command **trains all 43 trajectories**, evaluates the final selected models,
and exports their histories, settings, metrics and figures:

```bash
python -m campaign.reproduce all --root output/reproduction --archive data/wikiart.zip
```

The 43 trajectories comprise 19 screening arms, eight added-seed confirmation
arms, 12 constant-rate extensions, three paired decay extensions, and one separate
native128 demonstration. Native128 is excluded from the RGB64 ranking.
Use `--skip-native128` to omit that demonstration or `--skip-final-test` to defer
the final evaluation. These options apply to the `all` command.

The default `--selection-mode dynamic` selects from newly measured validation
outputs. `--selection-mode recorded` instead replays the published family,
epoch and raw/EMA opportunities, then ranks their newly measured confirmation
results. Recorded mode is an explicit replay, not a claim that those choices
would win a fresh search. Keep the same mode and output root across phases.

The default CUDA device is the appropriate setting for the supplied historical
CUDA training state. CPU inference/evaluation is available; a CUDA noise-generator
training state is not a device-independent random state. CPU training of the
legacy continuation would require a compatible original full100 state from an
original CPU run, rather than silently changing its sampling history.

## Run individual phases or resume training

The following sequence is equivalent to `all`. Supply the same `--archive`,
`--cache-dir`, `--selection-mode`, worker and device settings when overriding
defaults. An existing compatible RGB64 cache can be passed through `--cache-dir`.

```bash
python -m campaign.reproduce prepare --root output/reproduction
python -m campaign.reproduce screen --root output/reproduction
python -m campaign.reproduce select-screen --root output/reproduction
python -m campaign.reproduce confirm --root output/reproduction
python -m campaign.reproduce extend42 --root output/reproduction
python -m campaign.reproduce select-extension --root output/reproduction
python -m campaign.reproduce extend-confirm --root output/reproduction
python -m campaign.reproduce select-final --root output/reproduction
python -m campaign.reproduce evaluate --root output/reproduction
python -m campaign.reproduce native128 --root output/reproduction
python -m campaign.reproduce summarize --root output/reproduction
```

Fresh training uses a fixed two-epoch finite-optimization pilot, then resumes to
epoch200. The legacy arm starts from the historical full100 state, pilots through
epoch102, then continues to200. A repeated training phase resumes each run's own
full `last.pt`; partial epochs after its checkpoint are replayed. Milestone
`checkpoints/epoch_*.pt` files are inference-only and are never training parents.

Before extensions, the wrapper copies each same-seed full200 state into
`parents_epoch200/`, verifies its checksum and makes it read-only. Constant and
decay twins use that identical parent. Resuming an interrupted decay run uses its
own full state without applying the initial fork flag again.

Decisions are created once under `selection/`. Screening uses seed42 pooled FID
at100/150/200 and raw/EMA, with only150/200 available to the legacy continuation.
The top three fresh noncontrol families plus `C0F` are fixed before confirmation.
Seed42 extensions fix250/300 and raw/EMA before any added-seed extension starts.
The nine resulting specifications rank by mean FID over seeds123/314; seed42
does not enter that ranking mean. Each reported model uses the median-validation
seed among42/123/314, followed by one common held-out test evaluation.

## Evaluation without retraining

After `prepare`, this evaluates the three released selected checkpoints against
the fresh frozen plan, using an explicitly new evaluation directory:

```bash
python -m campaign.evaluate_selected \
  --selection results/improvement/selected_models.json \
  --cache-dir output/reproduction/cache/rgb64 \
  --plan output/reproduction/reference_plan/reference_plan.json \
  --feature-cache-dir output/reproduction/features \
  --output-dir output/released_model_evaluation
```

For a single checkpoint use `--checkpoint PATH --weights raw` or `--weights ema`
instead of `--selection`. `--reference validation_pool4000_a` selects validation
rather than final test. `--samples 0` omits generation; the default saves64 images
from common seed43 latent draws. Add `--dry-run` to inspect model specifications
without loading weights or the numerical libraries.

Final evaluation creates `evaluation_claim.json` before requesting test feature
caches and refuses to overwrite any evaluation directory. If interrupted,
preserve all files. Existing generated feature arrays can be scored explicitly:

```bash
python -m hw2_gan.evaluate_protocols score \
  --candidate PATH/generated_features.npz \
  --plan output/reproduction/reference_plan/reference_plan.json \
  --feature-cache-dir output/reproduction/features \
  --output PATH/metrics.json
```

Use that recovery only for an existing valid feature artifact whose metrics file
is absent. Do not remove a claim or silently repeat a final test. To continue
`all` after an already completed final evaluation, use `--skip-final-test`.

## Outputs and implementation checks

`analysis/` contains CSV histories and settings, the three selection freezes,
final metrics and, after successful final generation, the same four-panel figure
interface consumed by `hw2_gan.improvement_figures`. Its real-image row uses eight
actual training images from distinct styles; their indices and names are saved.
The real row can differ from the retained report row while the selected models
use the same declared seed43 prior convention.

`campaign/experiments.json` defines every tested parameter set. The historical
`selection_protocol*.json` documents are retained as provenance; the portable
workflow creates new path bindings and decisions rather than modifying them.
The old experiment-status fields describe preparation, not current execution.

Only `test_portable` is intended as a lightweight planning/metadata check. The
other supplied tests exercise numerical implementations and belong on a compute
host, including their CPU tests:

```bash
python -m campaign.tests.test_model
python -m campaign.tests.test_campaign
python -m campaign.tests.test_evaluate_protocols
python -m campaign.tests.test_diffaugment
python -m campaign.tests.test_data128
python -m campaign.tests.test_model128
```

They cover baseline equivalence, gradient isolation, EMA/buffer behavior,
checkpoint/resume state, reference/cache integrity, metric identities,
DiffAugment parity and native128 preprocessing/model shapes. Passing these tests
does not establish image quality or statistical significance.
