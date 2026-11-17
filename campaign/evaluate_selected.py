"""Evaluate explicit raw/EMA checkpoints under the common frozen protocol.

This is an evaluation entry point, not a selection procedure. A final-test
invocation creates its immutable claim before requesting any test feature cache.
If interrupted, preserve that directory and resume scoring saved feature arrays
explicitly; do not remove the claim and silently repeat model evaluation.
"""

import argparse
from pathlib import Path
import shlex
import sys

from .common import emit, freeze, load, sha256


def selection_models(path):
    value = load(path)
    # Both the portable final freeze and the released selected_models.json work.
    models = value.get("models", value)
    for name, spec in models.items():
        if not isinstance(spec, dict) or "checkpoint" not in spec:
            raise ValueError(f"Expected a named checkpoint specification: {name}")
        if spec.get("variant", "raw") not in ("raw", "ema"):
            raise ValueError(f"Unknown generator variant: {name}")
    return models


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--selection", type=Path,
                        help="Portable final_selection.json or released selected_models.json.")
    source.add_argument("--checkpoint", type=Path)
    parser.add_argument("--weights", choices=("raw", "ema"), default="raw")
    parser.add_argument("--expected-generator-state-sha256")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--feature-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference", default="final_test4000",
                        choices=("validation_pool4000_a", "final_test4000", "historical_test4096"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--sample-seed", type=int, default=43)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def evaluate(args):
    models = selection_models(args.selection) if args.selection else {
        "model": {"checkpoint": str(args.checkpoint), "variant": args.weights,
                  "generator_state_sha256": args.expected_generator_state_sha256}}
    if args.dry_run:
        for name, spec in models.items():
            emit("evaluate_plan", name=name, checkpoint=spec["checkpoint"],
                 variant=spec.get("variant", "raw"), reference=args.reference,
                 plan=str(args.plan), output=str(args.output_dir / name))
        return
    from hw2_gan.utils import require_compute, load_generator
    from hw2_gan.evaluate import _state_hash
    from hw2_gan.evaluate_protocols import (
        LEGACY, POOLED, load_reference_plan, precompute_real_features,
        extract_generator_features, score_feature_artifact,
    )
    import subprocess
    import torch

    require_compute()
    if args.batch_size < 1 or args.samples < 0:
        raise ValueError("Use a positive batch size and nonnegative sample count.")
    torch.set_num_threads(8)
    plan, _ = load_reference_plan(args.plan, cache_dir=args.cache_dir, verify_pixels=True)
    if args.selection:
        selected = load(args.selection)
        if selected.get("plan_sha256") not in (None, plan["plan_sha256"]):
            raise ValueError("Selection and evaluation plans differ.")
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh evaluation output directory; preserve prior/partial evaluations.")
    protocols = [LEGACY] if args.reference == "historical_test4096" else [LEGACY, POOLED]
    # Resolve and verify all model identities before the first test cache request.
    verified = {}
    for name, spec in models.items():
        checkpoint = Path(spec["checkpoint"]).resolve()
        digest = sha256(checkpoint)
        if spec.get("checkpoint_sha256") not in (None, digest):
            raise ValueError(f"Checkpoint checksum differs: {name}")
        generator, state = load_generator(checkpoint, torch.device(args.device),
                                          weights=spec.get("variant", "raw"))
        state_hash = _state_hash(generator)
        if spec.get("generator_state_sha256") not in (None, state_hash):
            raise ValueError(f"Generator tensors differ: {name}")
        if spec.get("epoch") not in (None, state["epoch"]):
            raise ValueError(f"Checkpoint epoch differs: {name}")
        verified[name] = dict(spec, checkpoint=str(checkpoint), checkpoint_sha256=digest,
                              epoch=state["epoch"], generator_state_sha256=state_hash)
        del generator, state
    args.output_dir.mkdir(parents=True)
    freeze(args.output_dir / "evaluation_claim.json", {
        "selection_sha256": sha256(args.selection) if args.selection else None,
        "plan_sha256": plan["plan_sha256"], "reference": args.reference,
        "models": verified,
        "policy": "Explicit frozen models; no tuning from final-test scores. Preserve this claim on failure.",
    })
    precompute_real_features(args.cache_dir, args.plan, args.feature_cache_dir,
                             [args.reference], protocol_ids=protocols,
                             device=args.device, batch_size=args.batch_size)
    results, evaluated_states = {}, {}
    for name, spec in verified.items():
        state_hash = spec["generator_state_sha256"]
        if state_hash in evaluated_states:
            previous = evaluated_states[state_hash]
            results[name] = dict(results[previous], specification=spec,
                                 reused_identical_generator=previous)
        else:
            generator, _ = load_generator(spec["checkpoint"], torch.device(args.device),
                                          weights=spec.get("variant", "raw"))
            destination = args.output_dir / name
            artifact = extract_generator_features(
                generator, args.cache_dir, args.plan, args.reference,
                args.feature_cache_dir, destination, protocol_ids=protocols,
                device=args.device, batch_size=args.batch_size,
                generator_variant=spec.get("variant", "raw"),
                checkpoint_metadata={"epoch": spec["epoch"], "checkpoint": spec["checkpoint"],
                                     "checkpoint_sha256": spec["checkpoint_sha256"]})
            metrics = score_feature_artifact(artifact, args.plan, args.feature_cache_dir,
                                             destination / "metrics.json")
            results[name] = {"specification": spec, "metrics": metrics["metrics"],
                             "metric_artifact": str(destination / "metrics.json")}
            evaluated_states[state_hash] = name
            del generator
        if args.samples:
            command = [sys.executable, "-m", "hw2_gan.generate", "--checkpoint", spec["checkpoint"],
                       "--weights", spec.get("variant", "raw"),
                       "--expected-generator-state-sha256", spec["generator_state_sha256"],
                       "--output-dir", str(args.output_dir / "generation" / name),
                       "--num-samples", str(args.samples), "--seed", str(args.sample_seed),
                       "--device", args.device]
            emit("command", argv=command, shell=shlex.join(command))
            subprocess.run(command, check=True)
    freeze(args.output_dir / "summary.json", {"models": results, "plan_sha256": plan["plan_sha256"]})


def main():
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()
