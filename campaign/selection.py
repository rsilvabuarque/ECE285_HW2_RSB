"""Selection from newly produced validation metrics, without model inference.

Seed 42 chooses family/epoch/variant. Seeds 123 and 314 rank nine fixed
specifications; their individual minima are never selected. Historical decisions
can instead be replayed explicitly using --selection-mode recorded.
"""

import csv
import math
from pathlib import Path

from .common import freeze, load, sha256

POOLED = "torchvision2048pool-v1"
LEGACY = "legacy1000prob-v1"


def history(run):
    with (Path(run) / "history.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {int(row["epoch"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Repeated history epoch: {run}")
    return result


def complete(run, target, train_count, legacy=False):
    rows = history(run)
    config = load(Path(run) / "config.json")
    if set(rows) != set(range(1, target + 1)):
        raise ValueError(f"Incomplete history through epoch {target}: {run}")
    if config.get("integration_smoke") or any(config.get(k) is not None for k in ("train_limit", "max_batches")):
        raise ValueError(f"Truncated/smoke training is ineligible: {run}")
    expected_steps = math.ceil(train_count / config["batch_size"])
    for epoch, row in rows.items():
        if legacy and epoch <= 100:
            continue
        if int(row["train_images"]) != train_count or int(row["g_steps"]) != expected_steps:
            raise ValueError(f"Incomplete data epoch: {run}, {epoch}")
        for key in ("g_loss", "d_loss", "g_gradient_norm", "seconds"):
            if not math.isfinite(float(row[key])):
                raise ValueError(f"Nonfinite {key}: {run}, {epoch}")
    return rows


def run_path(root, family, seed, schedule="base"):
    suffix = {"base": "", "constant_extension": "_constant", "linear_decay": "_decay"}[schedule]
    return Path(root) / "runs" / f"{family}{suffix}_s{seed}"


def record(root, plan, family, seed, schedule, epoch, variant):
    run = run_path(root, family, seed, schedule)
    row = history(run)[epoch]
    artifact = Path(row[f"val_{variant}_pooled_feature_artifact"])
    metric_path = artifact.with_name("metrics.json")
    metric = load(metric_path)
    reference = plan["references"]["validation_pool4000_a"]
    if (metric["plan_sha256"] != plan["plan_sha256"]
            or metric["generator_variant"] != variant
            or metric["checkpoint"]["epoch"] != epoch
            or metric["reference_name"] != "validation_pool4000_a"
            or metric["reference_indices"] != reference["global_indices"]
            or metric["prior_array_sha256"] != plan["prior_array_sha256"]["validation"]
            or Path(metric["checkpoint"]["run_directory"]).resolve() != run.resolve()):
        raise ValueError(f"Validation identity mismatch: {metric_path}")
    count = len(reference["global_indices"])
    for protocol in (POOLED, LEGACY):
        values = metric["metrics"][protocol]
        if values["num_real_images"] != count or values["num_generated_images"] != count:
            raise ValueError(f"Unequal reference/generated counts: {metric_path}")
    score = float(metric["metrics"][POOLED]["fid"])
    if not math.isfinite(score):
        raise ValueError(f"Nonfinite pooled FID: {metric_path}")
    field = "val_pool_fid" if variant == "raw" else "val_ema_pool_fid"
    if not row.get(field) or float(row[field]) != score:
        raise ValueError(f"History and metric disagree: {metric_path}")
    checkpoint = run / "checkpoints" / f"epoch_{epoch:03d}.pt"
    return {"family": family, "seed": seed, "schedule": schedule, "epoch": epoch,
            "variant": variant, "pooled_FID": score,
            "IS": metric["metrics"][LEGACY]["inception_score"],
            "probability_FID_N4000": metric["metrics"][LEGACY]["fid"],
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
            "generator_state_sha256": metric["generator_state_sha256"],
            "metric_artifact": str(metric_path), "metric_sha256": sha256(metric_path),
            "pooled_protocol_hash": metric["metrics"][POOLED]["protocol_hash"],
            "legacy_protocol_hash": metric["metrics"][LEGACY]["protocol_hash"]}


def candidates(root, plan, family, seed, schedule, epochs):
    return [record(root, plan, family, seed, schedule, epoch, variant)
            for epoch in epochs for variant in ("raw", "ema")]


def best(values):
    return min(values, key=lambda x: (x["pooled_FID"], x["epoch"], x["variant"] != "raw"))


def same_protocol(records):
    identities = {(r["pooled_protocol_hash"], r["legacy_protocol_hash"]) for r in records}
    if len(identities) != 1:
        raise ValueError("Candidates use different numerical evaluation protocols.")


def screen(root, plan_path, matrix, train_count, output, recorded=None):
    plan = load(plan_path)
    choices, all_values = [], []
    labels = [arm["id"] for arm in matrix["planned"]]
    for label in labels:
        complete(run_path(root, label, 42), 200, train_count, legacy=label == "C0legacy")
        values = candidates(root, plan, label, 42, "base",
                            (150, 200) if label == "C0legacy" else (100, 150, 200))
        all_values.extend(values)
        choices.append(best(values))
    same_protocol(all_values)
    order = {label: i for i, label in enumerate(labels)}
    if recorded is None:
        ranked = sorted((r for r in choices if r["family"] not in ("C0legacy", "C0F")),
                        key=lambda r: (r["pooled_FID"], order[r["family"]]))
        shortlist = ranked[:3] + [next(r for r in choices if r["family"] == "C0F")]
        leader = min(shortlist, key=lambda r: (r["pooled_FID"], order[r["family"]]))["family"]
    else:
        shortlist = [record(root, plan, r["family"], 42, "base", r["epoch"], r["variant"])
                     for r in recorded["shortlisted"]]
        leader = recorded["decay_leader"]
    return freeze(output, {"phase": "screening", "selection_mode": "recorded" if recorded else "dynamic",
                           "plan_sha256": plan["plan_sha256"], "matrix_sha256": sha256(Path(__file__).with_name("experiments.json")),
                           "per_family_choices": choices, "all_candidates": all_values,
                           "shortlisted": shortlist, "decay_leader": leader})


def extension(root, plan_path, screening_path, train_count, output, recorded=None):
    screening = load(screening_path)
    plan = load(plan_path)
    if screening["plan_sha256"] != plan["plan_sha256"]:
        raise ValueError("Screening and extension use different reference plans.")
    for seed in (123, 314):
        for spec in screening["shortlisted"]:
            for schedule in ("constant_extension", "linear_decay"):
                if (run_path(root, spec["family"], seed, schedule) / "history.csv").exists():
                    raise ValueError("Freeze seed42 extension choices before starting confirmation extensions.")
    specs, all_values = [], []
    arms = [(r["family"], "constant_extension") for r in screening["shortlisted"]]
    arms.append((screening["decay_leader"], "linear_decay"))
    for family, schedule in arms:
        complete(run_path(root, family, 42, schedule), 300, train_count)
        values = candidates(root, plan, family, 42, schedule, (250, 300))
        all_values.extend(values)
        if recorded is None:
            spec = best(values)
        else:
            old = next(r for r in recorded["extension_specs"] if r["family"] == family and r["schedule"] == schedule)
            spec = record(root, plan, family, 42, schedule, old["epoch"], old["variant"])
        specs.append(spec)
    same_protocol(all_values + screening["all_candidates"])
    return freeze(output, {"phase": "extension", "screening_sha256": sha256(screening_path),
                           "plan_sha256": plan["plan_sha256"],
                           "extension_specs": specs, "all_candidates": all_values})


def rank_specs(specs, seed_records, family_order):
    """Pure metadata operation, also used by the lightweight planning tests."""
    if len(specs) != len(seed_records):
        raise ValueError("Every fixed specification must have one seed-record group.")
    results = []
    for spec, records in zip(specs, seed_records):
        if len(records) != 3 or {r["seed"] for r in records} != {42, 123, 314}:
            raise ValueError("Every specification needs exactly the three declared seeds.")
        mean = sum(r["pooled_FID"] for r in records if r["seed"] in (123, 314)) / 2
        representative = sorted(records, key=lambda r: (r["pooled_FID"], r["seed"]))[1]
        results.append({**{k: spec[k] for k in ("family", "schedule", "epoch", "variant")},
                        "confirmation_mean_FID": mean, "seed_records": records,
                        "representative": representative})
    schedules = {"base": 0, "constant_extension": 1, "linear_decay": 2}
    return sorted(results, key=lambda r: (r["confirmation_mean_FID"], family_order[r["family"]],
                                          schedules[r["schedule"]], r["epoch"], r["variant"] != "raw"))


def final(root, plan_path, screening_path, extension_path, matrix, train_count,
          original_checkpoint, original_metadata, output):
    screening, extended = load(screening_path), load(extension_path)
    plan = load(plan_path)
    if extended["screening_sha256"] != sha256(screening_path) or plan["plan_sha256"] != screening["plan_sha256"]:
        raise ValueError("Selection parent or reference plan changed.")
    specs = screening["shortlisted"] + extended["extension_specs"]
    if len(specs) != 9:
        raise ValueError("Expected four base, four constant and one decay specifications.")
    records = []
    parents = []
    for spec in specs:
        values = []
        for seed in (42, 123, 314):
            run = run_path(root, spec["family"], seed, spec["schedule"])
            complete(run, 200 if spec["schedule"] == "base" else 300, train_count)
            values.append(record(root, plan, spec["family"], seed, spec["schedule"], spec["epoch"], spec["variant"]))
            if spec["schedule"] != "base":
                receipt = load(run / "continuation.json")
                parent = Path(receipt["immutable_parent"])
                if parent.stat().st_mode & 0o222 or sha256(parent) != receipt["parent_sha256"]:
                    raise ValueError("Full200 training parent changed or became writable.")
                parents.append(receipt)
        records.append(values)
    same_protocol([r for group in records for r in group])
    for seed in (42, 123, 314):
        twins = [p for p in parents if p["family"] == screening["decay_leader"] and p["seed"] == seed]
        if len(twins) != 2 or len({p["parent_sha256"] for p in twins}) != 1:
            raise ValueError("Constant and decay twins must share the identical full200 state.")
    order = {arm["id"]: i for i, arm in enumerate(matrix["planned"])}
    ranked = rank_specs(specs, records, order)
    control = next(r for r in ranked if r["family"] == "C0F")
    original = dict(original_metadata, checkpoint=str(Path(original_checkpoint).resolve()))
    if sha256(original_checkpoint) != original["checkpoint_sha256"]:
        raise ValueError("Original selected checkpoint changed.")
    return freeze(output, {"phase": "final_selection_before_test", "plan_sha256": plan["plan_sha256"],
                           "screening_sha256": sha256(screening_path), "extension_sha256": sha256(extension_path),
                           "ranked_nine_specs": ranked, "full200_parents": parents,
                           "models": {"original95": original, "matched_C0F": control["representative"],
                                      "winner": ranked[0]["representative"]},
                           "interpretation": "Ranks mean seeds123/314; seed42 chooses opportunities. Representatives are median validation seed of three; final test is not a three-seed mean."})
