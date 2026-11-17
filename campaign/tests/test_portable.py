"""Lightweight planning/metadata tests; no Torch, data cache or inference.

Any numerical values in these fixtures are synthetic and are not GAN results.
The existing scientific implementation checks are separate compute-host tests.
"""

import ast
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from campaign import reproduce, selection
from campaign.common import freeze, sha256
from campaign.evaluate_selected import build_parser as evaluation_parser, evaluate

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class PortablePlanning(unittest.TestCase):
    def test_complete_matrix_flags_match_actual_trainer_parser(self):
        tree = ast.parse((REPOSITORY / "hw2_gan/campaign_train.py").read_text())
        flags = {argument.value for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                 and node.func.attr == "add_argument" for argument in node.args
                 if isinstance(argument, ast.Constant) and isinstance(argument.value, str)}
        matrix = json.loads((REPOSITORY / "campaign/experiments.json").read_text())
        self.assertEqual(len(matrix["planned"]), 19)
        self.assertEqual(len({r["id"] for r in matrix["planned"]}), 19)
        for arm in matrix["planned"]:
            arguments = reproduce.parameter_arguments(arm["parameters"])
            self.assertTrue({a for a in arguments if a.startswith("--")} <= flags)
            self.assertEqual(arm["parameters"]["data_order"],
                             "legacy" if arm["id"] == "C0legacy" else "epoch-v1")
        diffaug = next(r for r in matrix["planned"] if r["id"] == "DiffAug")
        self.assertEqual(diffaug["parameters"]["diffaugment_policy"], "color,translation,cutout")

    def test_all_dry_run_has_43_trajectories_and_one_final_evaluation(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            base = Path(temporary)
            recorded = base / "recorded"
            shortlist = [{"family": name, "epoch": 200, "variant": "ema", "schedule": "base"}
                         for name in ("A1", "G1", "T3", "C0F")]
            write_json(recorded / "screening_freeze.json", {"shortlisted": shortlist, "decay_leader": "A1"})
            write_json(recorded / "extension_freeze.json", {"extension_specs": []})
            output = base / "fresh"
            args = reproduce.build_parser().parse_args([
                "all", "--root", str(output), "--recorded-results", str(recorded), "--dry-run"])
            capture = io.StringIO()
            with redirect_stdout(capture):
                reproduce.Workflow(args).execute()
            events = [json.loads(line) for line in capture.getvalue().splitlines()]
            commands = [e["argv"] for e in events if e["event"] == "command"]
            training = [a for a in commands if a[2] == "hw2_gan.campaign_train"]
            outputs = {a[a.index("--output-dir") + 1] for a in training}
            self.assertEqual(len(outputs), 43)
            self.assertEqual(len(training), 71)  # 28 pilot/full pairs and 15 continuations.
            final = [a for a in commands if a[2] == "campaign.evaluate_selected" and "--selection" in a]
            self.assertEqual(len(final), 1)
            self.assertFalse(output.exists())
            legacy = [a for a in training if a[a.index("--output-dir") + 1].endswith("C0legacy_s42")]
            self.assertEqual([a[a.index("--epochs") + 1] for a in legacy], ["102", "200"])
            self.assertIn("--resume", legacy[0])
            decay = [a for a in training if "--fork-linear-decay" in a]
            self.assertEqual(len(decay), 3)
            self.assertTrue(all("parents_epoch200" in a[a.index("--resume") + 1] for a in decay))
            phases = [e["name"] for e in events if e["event"] == "phase"]
            self.assertLess(phases.index("select-screen"), phases.index("confirm"))
            self.assertLess(phases.index("select-extension"), phases.index("extend-confirm"))
            self.assertLess(phases.index("select-final"), phases.index("evaluate"))

    def test_ranking_excludes_discovery_seed_and_uses_median_representative(self):
        # Synthetic records: an excellent discovery seed must not win the mean.
        specs = [{"family": name, "schedule": "base", "epoch": 200, "variant": "ema"}
                 for name in ("A1", "C0F")]
        records = [[{"seed": seed, "pooled_FID": score} for seed, score in pairs]
                   for pairs in (((42, 0), (123, 12), (314, 14)),
                                 ((42, 1000), (123, 10), (314, 11))) ]
        ranked = selection.rank_specs(specs, records, {"A1": 0, "C0F": 1})
        self.assertEqual(ranked[0]["family"], "C0F")
        self.assertEqual(ranked[0]["confirmation_mean_FID"], 10.5)
        self.assertEqual(ranked[0]["representative"]["seed"], 314)
        self.assertEqual(ranked[1]["representative"]["seed"], 123)

    def test_freeze_rejects_changed_decision(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            path = Path(temporary) / "freeze.json"
            freeze(path, {"value": 1})
            freeze(path, {"value": 1})
            with self.assertRaises(FileExistsError):
                freeze(path, {"value": 2})

    def test_record_checks_reference_prior_variant_and_history(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            root = Path(temporary)
            run = selection.run_path(root, "A1", 42)
            artifact = run / "validation/generated_features.npz"
            metric_path = artifact.with_name("metrics.json")
            checkpoint = run / "checkpoints/epoch_200.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"synthetic checkpoint fixture, not model weights")
            plan = {"plan_sha256": "plan", "prior_array_sha256": {"validation": "prior"},
                    "references": {"validation_pool4000_a": {"global_indices": [4, 8]}}}
            metric = {"plan_sha256": "plan", "generator_variant": "ema", "checkpoint": {"epoch": 200, "run_directory": str(run)},
                      "reference_name": "validation_pool4000_a", "reference_indices": [4, 8],
                      "prior_array_sha256": "prior", "generator_state_sha256": "synthetic",
                      "metrics": {name: {"num_real_images": 2, "num_generated_images": 2,
                                         "fid": 12.5, "inception_score": 2.0, "protocol_hash": name}
                                  for name in (selection.POOLED, selection.LEGACY)}}
            write_json(metric_path, metric)
            with (run / "history.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, ["epoch", "val_ema_pooled_feature_artifact", "val_ema_pool_fid"])
                writer.writeheader()
                writer.writerow({"epoch": 200, "val_ema_pooled_feature_artifact": str(artifact), "val_ema_pool_fid": 12.5})
            value = selection.record(root, plan, "A1", 42, "base", 200, "ema")
            self.assertEqual(value["checkpoint_sha256"], sha256(checkpoint))
            metric["prior_array_sha256"] = "different"
            write_json(metric_path, metric)
            with self.assertRaises(ValueError):
                selection.record(root, plan, "A1", 42, "base", 200, "ema")

    def test_evaluation_dry_run_needs_no_weights_or_torch(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            base = Path(temporary)
            spec = base / "selection.json"
            write_json(spec, {"models": {"winner": {"checkpoint": "missing.pt", "variant": "ema"}}})
            args = evaluation_parser().parse_args([
                "--selection", str(spec), "--cache-dir", str(base / "cache"),
                "--plan", str(base / "plan.json"), "--feature-cache-dir", str(base / "features"),
                "--output-dir", str(base / "output"), "--dry-run"])
            with redirect_stdout(io.StringIO()):
                evaluate(args)
            self.assertFalse((base / "output").exists())


if __name__ == "__main__":
    unittest.main()
