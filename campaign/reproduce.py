"""Portable, sequential reproduction of the supplementary GAN campaign.

No scheduler is invoked. Run numerical phases on a suitable GPU/compute host.
The default dynamic mode selects from fresh validation outputs. Recorded mode
replays the published family/epoch/variant opportunities, while measuring new
outputs; it does not claim that the replay would win a fresh dynamic search.
All phases are restartable from completed full checkpoints. A final evaluation
is deliberately once-only: preserve a partial evaluation and audit recovery.
"""

import argparse
import csv
import math
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from .common import emit, freeze, load, sha256
from . import selection

REPOSITORY = Path(__file__).resolve().parents[1]
BOOLEAN_PARAMETERS = {"d_spectral_norm", "d_no_batchnorm", "d_eval_for_g", "horizontal_flip"}
PHASES = ("prepare", "screen", "select-screen", "confirm", "extend42", "select-extension",
          "extend-confirm", "select-final", "evaluate", "native128", "summarize", "all")


def parameter_arguments(parameters):
    """Translate every recorded matrix setting without changing its meaning."""
    arguments = []
    for key, value in parameters.items():
        flag = "--" + key.replace("_", "-")
        if key in BOOLEAN_PARAMETERS:
            if value:
                arguments.append(flag)
        elif value != "":
            arguments.extend((flag, str(value)))
    return arguments


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=PHASES)
    parser.add_argument("--root", type=Path, required=True, help="A new output root, outside retained results.")
    parser.add_argument("--archive", type=Path, default=Path("data/wikiart.zip"))
    parser.add_argument("--cache-dir", type=Path, help="An existing RGB64 cache, or ROOT/cache/rgb64 by default.")
    parser.add_argument("--original-full100", type=Path, default=Path("checkpoints/original_epoch100_full.pt"))
    parser.add_argument("--original95", type=Path, default=Path("results/improvement/checkpoints/original95.pt"))
    parser.add_argument("--recorded-results", type=Path, default=Path("results/improvement"))
    parser.add_argument("--selection-mode", choices=("dynamic", "recorded"), default="dynamic")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--workers", type=int, default=8, help="Image-preparation workers.")
    parser.add_argument("--num-workers", type=int, default=4, help="Fresh-arm training loader workers.")
    parser.add_argument("--skip-native128", action="store_true", help="In all, omit the separately ranked native-resolution demo.")
    parser.add_argument("--skip-final-test", action="store_true", help="In all, stop short of final-test evaluation.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands/decisions only; do not create files or load Torch.")
    return parser


class Workflow:
    def __init__(self, args):
        self.args = args
        self.root = args.root.resolve()
        self.cache = (args.cache_dir or self.root / "cache/rgb64").resolve()
        self.plan = self.root / "reference_plan/reference_plan.json"
        self.features = self.root / "features"
        self.screening = self.root / "selection/screening.json"
        self.extended = self.root / "selection/extension.json"
        self.final = self.root / "selection/final_selection.json"
        self.recorded = args.recorded_results.resolve()
        self.matrix = load(Path(__file__).with_name("experiments.json"))
        self.arms = {arm["id"]: arm for arm in self.matrix["planned"]}
        if len(self.arms) != 19:
            raise ValueError("Expected the complete nineteen-family experiment matrix.")
        self._planned_screening = None
        self._planned_extension = None

    def command(self, module, *arguments):
        argv = [sys.executable, "-m", module, *map(str, arguments)]
        emit("command", argv=argv, shell=shlex.join(argv))
        if not self.args.dry_run:
            subprocess.run(argv, cwd=REPOSITORY, check=True)

    def configure(self):
        if self.args.dry_run:
            return
        from hw2_gan.utils import require_compute
        require_compute()
        if self.root == REPOSITORY or self.recorded == self.root or self.recorded in self.root.parents:
            raise ValueError("Use a fresh output root outside the retained result directory.")
        freeze(self.root / "workflow.json", {
            "schema": "portable-hw2-reproduction-v1", "root": str(self.root),
            "cache": str(self.cache), "original_full100": str(self.args.original_full100.resolve()),
            "original95": str(self.args.original95.resolve()), "selection_mode": self.args.selection_mode,
            "device": self.args.device, "num_workers": self.args.num_workers,
            "matrix_sha256": sha256(Path(__file__).with_name("experiments.json")),
            "original_full100_sha256": sha256(self.args.original_full100.resolve()),
            "original95_sha256": sha256(self.args.original95.resolve()),
            "scientific_modules_sha256": {p.name: sha256(p) for p in sorted((REPOSITORY / "hw2_gan").glob("*.py"))},
            "portable_modules_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        })

    def train_count(self):
        return len(load(self.cache / "manifest.json")["splits"]["train"])

    def prepare(self):
        if self.args.dry_run or not (self.cache / "manifest.json").exists():
            self.command("hw2_gan.data", "--archive", self.args.archive.resolve(),
                         "--output-dir", self.cache, "--workers", self.args.workers, "--seed", 42)
        if self.args.dry_run or not self.plan.exists():
            pixel_hash = "MEASURED_IMAGES_NPY_SHA256" if self.args.dry_run else sha256(self.cache / "images.npy")
            self.command("hw2_gan.evaluate_protocols", "freeze-plan", "--cache-dir", self.cache,
                         "--output-dir", self.plan.parent,
                         "--historical-test-metrics", self.recorded / "historical_test4096_metrics.json",
                         "--pixel-cache-sha256", pixel_hash)
        self.command("hw2_gan.evaluate_protocols", "precompute", "--cache-dir", self.cache,
                     "--plan", self.plan, "--feature-cache-dir", self.features,
                     "--references", "validation_legacy1024", "validation_pool4000_a", "validation_pool4000_b",
                     "--device", self.args.device)
        context = self.root / "validation_context/real_vs_real.json"
        if self.args.dry_run or not context.exists():
            self.command("hw2_gan.evaluate_protocols", "real-control", "--plan", self.plan,
                         "--feature-cache-dir", self.features, "--output", context, "--device", self.args.device)
        baseline = self.root / "validation_context/original95"
        if self.args.dry_run or not (baseline / "summary.json").exists():
            self.command("campaign.evaluate_selected", "--checkpoint", self.args.original95.resolve(),
                         "--cache-dir", self.cache, "--plan", self.plan, "--feature-cache-dir", self.features,
                         "--reference", "validation_pool4000_a", "--output-dir", baseline,
                         "--samples", 0, "--device", self.args.device)

    def training_arguments(self, family, seed):
        run = selection.run_path(self.root, family, seed)
        parameters = dict(self.arms[family]["parameters"], seed=seed)
        arguments = ["--cache-dir", str(self.cache), "--output-dir", str(run),
                     "--evaluation-plan", str(self.plan), "--feature-cache", str(self.features),
                     "--eval-every", "25", "--device", self.args.device]
        # Historical full100 fixes its original loader worker count to four.
        arguments += ["--num-workers", "4" if family == "C0legacy" else str(self.args.num_workers)]
        return run, arguments + parameter_arguments(parameters)

    def train_to(self, output, arguments, target, initial_resume=None, pilot=2):
        """Resume only completed full states; interrupted partial epochs replay."""
        output = Path(output)
        last = output / "last.pt"
        epoch = 0
        resume = initial_resume
        if not self.args.dry_run:
            import torch
            if last.exists():
                state = torch.load(last, map_location="cpu", weights_only=True)
                epoch, resume = state["epoch"], last
                del state
            elif initial_resume:
                state = torch.load(initial_resume, map_location="cpu", weights_only=True)
                if state["epoch"] != 100 or not {"discriminator_state", "g_optimizer_state", "d_optimizer_state", "data_rng_state", "noise_rng_state"} <= state.keys():
                    raise ValueError("C0legacy requires the original full epoch100 state, not inference weights.")
                if state["config"].get("device", "cuda").split(":")[0] != self.args.device:
                    raise ValueError("Legacy continuation must preserve its training RNG device; the supplied full100 state uses CUDA.")
                epoch = state["epoch"]
                del state
            if epoch >= target:
                emit("completed_training_reused", output=str(output), epoch=epoch)
                return
            if output.exists() and any(output.iterdir()) and not last.exists():
                raise FileExistsError(f"Partial output has no full checkpoint; preserve it and inspect: {output}")
        first_target = (102 if initial_resume else pilot)
        if self.args.dry_run or epoch < first_target:
            command = arguments + ["--epochs", str(first_target)]
            if resume:
                command += ["--resume", str(resume)]
            self.command("hw2_gan.campaign_train", *command)
            resume, epoch = last, first_target
        if not self.args.dry_run:
            rows = selection.history(output)
            pilot_start = 101 if initial_resume else 1
            for completed in range(pilot_start, first_target + 1):
                row = rows[completed]
                for key in ("g_loss", "d_loss", "g_gradient_norm", "seconds"):
                    if not row.get(key) or not math.isfinite(float(row[key])):
                        raise ValueError(f"Pilot has a nonfinite/missing {key}: {output}, epoch{completed}")
        if epoch < target:
            self.command("hw2_gan.campaign_train", *arguments, "--epochs", target, "--resume", resume)

    def screen(self):
        for family in self.arms:
            output, arguments = self.training_arguments(family, 42)
            parent = self.args.original_full100.resolve() if family == "C0legacy" else None
            self.train_to(output, arguments, 200, initial_resume=parent)

    def screening_info(self):
        if self.screening.exists():
            return load(self.screening)
        if self.args.dry_run:
            if self._planned_screening is None:
                emit("planning_assumption", message="Dynamic selection is unavailable before fresh training. Dry-run topology uses recorded shortlist only; actual dynamic run uses new scores.")
                self._planned_screening = load(self.recorded / "screening_freeze.json")
            return self._planned_screening
        raise FileNotFoundError("Run select-screen before confirmation or extension phases.")

    def select_screen(self):
        if self.screening.exists():
            return self.screening_info()
        if self.args.dry_run:
            emit("selection_plan", phase="screening", mode=self.args.selection_mode,
                 output=str(self.screening), candidates="100/150/200 raw/EMA; legacy150/200",
                 rule="top3 fresh noncontrol plus mandatory C0F")
            return self.screening_info()
        if any((selection.run_path(self.root, family, seed) / "history.csv").exists()
               for family in self.arms for seed in (123, 314)):
            raise ValueError("Freeze screening before starting any confirmation trajectories.")
        recorded = load(self.recorded / "screening_freeze.json") if self.args.selection_mode == "recorded" else None
        return selection.screen(self.root, self.plan, self.matrix, self.train_count(), self.screening, recorded)

    def confirm(self):
        for seed in (123, 314):
            for spec in self.screening_info()["shortlisted"]:
                output, arguments = self.training_arguments(spec["family"], seed)
                self.train_to(output, arguments, 200)

    def preserve_parent(self, family, seed):
        source = selection.run_path(self.root, family, seed) / "last.pt"
        destination = self.root / "parents_epoch200" / f"{family}_s{seed}" / "last.pt"
        if self.args.dry_run:
            emit("preserve_full200_plan", source=str(source), immutable_parent=str(destination))
            return destination, "PARENT_SHA256"
        import os
        import tempfile
        import torch
        state = torch.load(source, map_location="cpu", weights_only=True)
        required = {"generator_state", "generator_ema_state", "discriminator_state", "g_optimizer_state",
                    "d_optimizer_state", "data_rng_state", "worker_rng_state", "noise_rng_state",
                    "augmentation_rng_state", "torch_rng_state", "cuda_rng_states", "python_rng_state",
                    "numpy_rng_state", "history", "global_g_steps", "global_d_steps", "cache_fingerprint"}
        if (state["epoch"] != 200 or state["config"]["seed"] != seed or not required <= state.keys()
                or state["config"].get("learning_rate_schedule", "constant") != "constant"):
            raise ValueError("Continuation needs the same-seed complete constant-rate epoch200 state.")
        del state
        digest = sha256(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            descriptor, temporary = tempfile.mkstemp(dir=destination.parent, prefix=".parent_")
            os.close(descriptor)
            try:
                shutil.copyfile(source, temporary)
                os.chmod(temporary, 0o444)
                os.link(temporary, destination)
            finally:
                Path(temporary).unlink(missing_ok=True)
        if sha256(destination) != digest or destination.stat().st_mode & 0o222:
            raise ValueError("Full200 parent changed or became writable.")
        freeze(destination.parent / "parent.json", {"family": family, "seed": seed,
                                                    "source": str(source), "sha256": digest})
        return destination, digest

    def continue_family(self, family, seed, schedule):
        parent, digest = self.preserve_parent(family, seed)
        output = selection.run_path(self.root, family, seed, schedule)
        arguments = ["--cache-dir", str(self.cache), "--output-dir", str(output),
                     "--epochs", "300", "--resume", str(parent),
                     "--device", self.args.device,
                     "--evaluation-plan", str(self.plan), "--feature-cache", str(self.features),
                     "--pooled-epochs", "250,300", "--diagnostic-epochs", "250,300"]
        if schedule == "linear_decay":
            arguments.append("--fork-linear-decay")
        if not self.args.dry_run and (output / "last.pt").exists():
            import torch
            state = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
            epoch = state["epoch"]
            del state
            if epoch >= 300:
                emit("completed_continuation_reused", output=str(output), epoch=epoch)
            else:
                # Fork flag applies only on the initial200 clone, never its own resume.
                arguments = ["--cache-dir", str(self.cache), "--output-dir", str(output),
                             "--epochs", "300", "--resume", str(output / "last.pt"),
                             "--device", self.args.device]
                self.command("hw2_gan.campaign_train", *arguments)
        else:
            self.command("hw2_gan.campaign_train", *arguments)
        if not self.args.dry_run:
            if sha256(parent) != digest:
                raise ValueError("Continuation changed its full200 parent.")
            freeze(output / "continuation.json", {"family": family, "seed": seed, "schedule": schedule,
                                                  "immutable_parent": str(parent), "parent_sha256": digest,
                                                  "screening_sha256": sha256(self.screening),
                                                  "extension_sha256": sha256(self.extended) if seed != 42 else None})

    def extend(self, seeds):
        screening = self.screening_info()
        for seed in seeds:
            for spec in screening["shortlisted"]:
                self.continue_family(spec["family"], seed, "constant_extension")
            self.continue_family(screening["decay_leader"], seed, "linear_decay")

    def select_extension(self):
        if self.extended.exists():
            return load(self.extended)
        if self.args.dry_run:
            emit("selection_plan", phase="extension", output=str(self.extended),
                 rule="seed42 freezes250/300 raw/EMA for4constant+1decay before confirmation extensions")
            return load(self.recorded / "extension_freeze.json")
        recorded = load(self.recorded / "extension_freeze.json") if self.args.selection_mode == "recorded" else None
        return selection.extension(self.root, self.plan, self.screening, self.train_count(), self.extended, recorded)

    def select_final(self):
        if self.final.exists():
            return load(self.final)
        if self.args.dry_run:
            emit("selection_plan", phase="final", output=str(self.final),
                 rule="rank9fixedspecs bymeanFIDseeds123/314; representative medianvalidationseed42/123/314; no test access")
            return None
        original = load(self.recorded / "selected_models.json")["original95"]
        return selection.final(self.root, self.plan, self.screening, self.extended,
                               self.matrix, self.train_count(), self.args.original95.resolve(), original, self.final)

    def evaluate(self):
        self.command("campaign.evaluate_selected", "--selection", self.final,
                     "--cache-dir", self.cache, "--plan", self.plan, "--feature-cache-dir", self.features,
                     "--output-dir", self.root / "final_evaluation", "--device", self.args.device)

    def native128(self):
        cache = self.root / "cache/rgb128"
        plan = self.root / "native128/reference_plan/reference_plan.json"
        features = self.root / "native128/features"
        if self.args.dry_run or not (cache / "manifest.json").exists():
            self.command("hw2_gan.data128", "--archive", self.args.archive.resolve(),
                         "--reference-manifest", self.cache / "manifest.json",
                         "--output-dir", cache, "--workers", self.args.workers)
        if self.args.dry_run or not plan.exists():
            pixel_hash = "MEASURED_NATIVE128_IMAGES_NPY_SHA256" if self.args.dry_run else sha256(cache / "images.npy")
            self.command("hw2_gan.evaluate_protocols", "freeze-plan", "--cache-dir", cache,
                         "--output-dir", plan.parent, "--historical-test-metrics",
                         self.recorded / "historical_test4096_metrics.json", "--pixel-cache-sha256", pixel_hash)
        self.command("hw2_gan.evaluate_protocols", "precompute", "--cache-dir", cache, "--plan", plan,
                     "--feature-cache-dir", features, "--references", "validation_legacy1024", "validation_pool4000_a",
                     "--device", self.args.device)
        output = self.root / "runs/RGB128demo_s42"
        arguments = ["--cache-dir", str(cache), "--output-dir", str(output), "--image-size", "128",
                     "--base-channels", "32", "--batch-size", "128", "--latent-dim", "100",
                     "--num-workers", str(self.args.num_workers), "--data-order", "epoch-v1",
                     "--ema-decay", ".999", "--seed", "42", "--evaluation-plan", str(plan),
                     "--feature-cache", str(features), "--pooled-epochs", "100",
                     "--diagnostic-epochs", "25,50,100", "--device", self.args.device]
        self.train_to(output, arguments, 100)

    def summarize(self):
        if self.args.dry_run:
            emit("summarize_plan", output=str(self.root / "analysis"), content="all histories, model settings, candidate scores and final test metrics")
            return
        out = self.root / "analysis"
        if out.exists():
            raise FileExistsError("Preserve the prior analysis directory; choose explicit output/version changes.")
        (out / "histories").mkdir(parents=True)
        (out / "configs").mkdir()
        all_rows, runs = [], []
        for run in sorted((self.root / "runs").iterdir()):
            if not (run / "history.csv").exists():
                continue
            rows = list(selection.history(run).values())
            config = load(run / "config.json")
            shutil.copyfile(run / "history.csv", out / "histories" / f"{run.name}.csv")
            shutil.copyfile(run / "config.json", out / "configs" / f"{run.name}.json")
            all_rows.extend(dict(run=run.name, **row) for row in rows)
            runs.append({"run": run.name, "seed": config["seed"], "epochs": int(rows[-1]["epoch"]),
                         "image_size": config["image_size"], "G_width": config["g_base_channels"],
                         "D_width": config["d_base_channels"], "real_horizontal_flip": config["horizontal_flip"],
                         "schedule": config["learning_rate_schedule"], "last_raw_pooled_FID": rows[-1].get("val_pool_fid"),
                         "last_EMA_pooled_FID": rows[-1].get("val_ema_pool_fid")})
        for filename, values in (("all_epoch_metrics.csv", all_rows), ("all_runs.csv", runs)):
            fields = list(dict.fromkeys(key for value in values for key in value))
            with (out / filename).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fields); writer.writeheader(); writer.writerows(values)
        for source in (self.screening, self.extended, self.final):
            shutil.copyfile(source, out / source.name)
        if (self.root / "final_evaluation/summary.json").exists():
            shutil.copyfile(self.root / "final_evaluation/summary.json", out / "final_metrics.json")
            selected = load(self.final)["models"]
            freeze(out / "figure_config.json", {
                "winner_family": selected["winner"]["family"],
                "winner_variant": selected["winner"]["variant"],
                "models": {name: {key: spec[key] for key in ("epoch", "variant", "seed")}
                           for name, spec in selected.items()},
                "prior_seed": 43, "displayed_tile_indices": list(range(8)), "full_fixed_grid_count": 64,
            })
            shutil.copytree(self.root / "final_evaluation/generation", out / "generation")
            # Show eight actual training images, one per style, at native size.
            import errno
            import numpy as np
            from PIL import Image
            manifest = load(self.cache / "manifest.json")
            seen, indices = set(), []
            for index in manifest["splits"]["train"]:
                style = Path(manifest["files"][index]).parts[0]
                if style not in seen:
                    seen.add(style); indices.append(index)
                if len(indices) == 8:
                    break
            if len(indices) != 8:
                raise ValueError("The comparison panel expects eight training styles.")
            try:
                pixels = np.load(self.cache / "images.npy", mmap_mode="r")
            except OSError as error:
                if error.errno not in (errno.ENOSYS, errno.ENODEV, errno.EOPNOTSUPP):
                    raise
                pixels = np.load(self.cache / "images.npy")
            real = out / "generation/real"
            real.mkdir()
            for column, index in enumerate(indices):
                Image.fromarray(pixels[index]).save(real / f"sample_{column:05d}.png")
            freeze(out / "real_training_inputs.json", {"global_indices": indices,
                                                         "files": [manifest["files"][i] for i in indices]})
            self.command("hw2_gan.improvement_figures", "--results-dir", out, "--output-dir", out / "figures")
        emit("summary_complete", runs=len(runs), output=str(out))

    def execute(self):
        self.configure()
        phase = self.args.phase
        if phase == "all":
            phases = list(PHASES[:8])
            if not self.args.skip_final_test:
                phases.append("evaluate")
            if not self.args.skip_native128:
                phases.append("native128")
            phases.append("summarize")
        else:
            phases = [phase]
        for current in phases:
            emit("phase", name=current, dry_run=self.args.dry_run)
            if current == "extend42":
                self.extend((42,))
            elif current == "extend-confirm":
                if not self.args.dry_run and not self.extended.exists():
                    raise FileNotFoundError("Freeze extension choices before confirmation extensions.")
                self.extend((123, 314))
            else:
                getattr(self, current.replace("-", "_"))()


def main():
    Workflow(build_parser().parse_args()).execute()


if __name__ == "__main__":
    main()
