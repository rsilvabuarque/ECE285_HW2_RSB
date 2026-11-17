"""Rebuild the final report's figures/tables from frozen measurements and pixels.

This offline analysis does not import Torch or run a model. The original training
histories remain in results/; the supplementary measurements copied into
report/inputs/ retain their source bytes and a SHA256 inventory.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
POOLED = "torchvision2048pool-v1"


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def mean_fid(rows, family, epoch, schedule="base"):
    values = []
    for seed in (123, 314):
        name = f"{family}_s{seed}" if schedule == "base" else f"{family}_{schedule}_s{seed}"
        matches = [r for r in rows if r["run"] == name and int(r["epoch"]) == epoch]
        if len(matches) != 1 or not matches[0]["val_ema_pool_fid"]:
            raise ValueError(f"Expected one pooled EMA measurement: {name}, epoch {epoch}")
        values.append(float(matches[0]["val_ema_pool_fid"]))
    return {"mean": float(np.mean(values)), "seeds": [123, 314], "values": values}


def table(path, header, rows, alignment):
    content = [r"\begin{tabular}{" + alignment + "}", r"\toprule",
               " & ".join(header) + r" \\", r"\midrule"]
    content.extend(" & ".join(row) + r" \\" for row in rows)
    content.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(content) + "\n")


def build(repository, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    repository, output = Path(repository), Path(output)
    inputs = repository / "report/inputs/improvement"
    for name, expected in read_json(inputs / "SOURCE_FILES.json")["files"].items():
        actual = hashlib.sha256((inputs / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen input changed: {name}")
    figures, tables = output / "figures", output / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7})

    selection = read_json(repository / "results/selected/selection.json")
    original_test = read_json(repository / "results/final_evaluation/metrics.json")
    for arm in selection["arms"]:
        folder = "selected" if arm["run"] == "baseline64" else "lowerlr64"
        for name in ("history", "config"):
            suffix = "csv" if name == "history" else "json"
            path = repository / "results" / folder / f"{name}.{suffix}"
            if hashlib.sha256(path.read_bytes()).hexdigest() != arm[f"{name}_sha256"]:
                raise ValueError(f"Original {name} changed: {folder}")
    baseline_rows = []
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 1.55))
    for folder, color, label in [("selected", "#2166ac", r"$2\times10^{-4}$"),
                                 ("lowerlr64", "#b35806", r"$10^{-4}$")]:
        history = read_csv(repository / "results" / folder / "history.csv")
        if [int(r["epoch"]) for r in history] != list(range(1, 101)):
            raise ValueError("Original comparison requires both complete 100-epoch histories")
        for ax, field in zip(axes, ("g_loss", "d_loss", "val_fid")):
            measured = [r for r in history if r[field]]
            ax.plot([int(r["epoch"]) for r in measured], [float(r[field]) for r in measured],
                    color=color, lw=.8, marker="o" if field == "val_fid" else None,
                    ms=2, label=label)
    for ax, ylabel in zip(axes, ("Generator BCE", "Discriminator BCE", "Probability-FID")):
        ax.axvline(selection["selected_epoch"], color=".4", ls="--", lw=.7)
        ax.set(xlabel="Epoch", ylabel=ylabel)
        ax.grid(alpha=.2)
    axes[2].legend(frameon=False, fontsize=6)
    fig.tight_layout(pad=.4, w_pad=.65)
    fig.savefig(figures / "baseline_training.png", dpi=240)
    plt.close(fig)
    for arm in selection["arms"]:
        rate = r"$2\times10^{-4}$" if arm["learning_rate"] == .0002 else r"$10^{-4}$"
        baseline_rows.append([rate, str(arm["best_epoch"]),
                              f'{arm["validation_probability_fid"]:.6f}',
                              f'{arm["validation_inception_score"]:.2f}'])
    table(tables / "baseline.tex", ["Learning rate", "Selected epoch", r"Probability-FID $\downarrow$", r"IS $\uparrow$"],
          baseline_rows, "lrrr")

    rows = read_csv(inputs / "all_epoch_metrics.csv")
    trajectories = {}
    fig, ax = plt.subplots(figsize=(6.5, 1.6))
    for family, color, label in [("C0F", "#666666", "Unaugmented, constant rate"),
                                 ("A1", "#2166ac", "Horizontal flips, constant rate")]:
        points = [mean_fid(rows, family, e) for e in (100, 150, 200)]
        points += [mean_fid(rows, family, e, "constant") for e in (250, 300)]
        epochs = [100, 150, 200, 250, 300]
        trajectories[family] = {str(e): p for e, p in zip(epochs, points)}
        for index in range(2):
            ax.plot(epochs, [p["values"][index] for p in points], color=color, alpha=.22, lw=.65)
        ax.plot(epochs, [p["mean"] for p in points], "o-", color=color, lw=1.2, ms=3, label=label)
    decay = [mean_fid(rows, "A1", 200)] + [mean_fid(rows, "A1", e, "decay") for e in (250, 300)]
    trajectories["A1_decay"] = {str(e): p for e, p in zip((200, 250, 300), decay)}
    for index in range(2):
        ax.plot([200, 250, 300], [p["values"][index] for p in decay], color="#b35806", alpha=.22, lw=.65)
    ax.plot([200, 250, 300], [p["mean"] for p in decay], "s-", color="#b35806", lw=1.2, ms=3,
            label="Horizontal flips, late rate decay")
    ax.axvline(200, color=".5", ls=":", lw=.8)
    ax.set(xlabel="Epoch", ylabel="Validation FID", xticks=[100, 150, 200, 250, 300])
    ax.grid(alpha=.2)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.subplots_adjust(left=.105, right=.70, bottom=.29, top=.96)
    fig.savefig(figures / "refinement_training.png", dpi=240)
    plt.close(fig)
    refinement_specs = [("Unaugmented + EMA", "C0F", 200, "base"),
                        ("Horizontal flips + EMA", "A1", 200, "base"),
                        ("Flips, constant rate + EMA", "A1", 300, "constant"),
                        ("Flips, late rate decay + EMA", "A1", 300, "decay")]
    refinement = [{"recipe": label, "epoch": epoch, **mean_fid(rows, family, epoch, schedule)}
                  for label, family, epoch, schedule in refinement_specs]
    table(tables / "refinement.tex", ["Training recipe", "Epoch", r"Mean validation FID $\downarrow$"],
          [[r["recipe"], str(r["epoch"]), f'{r["mean"]:.2f}'] for r in refinement], "lrr")

    names = [("original95", "Original model", "95 / ordinary"),
             ("matched_C0F", "Fresh unaugmented control", "250 / EMA"),
             ("winner", "Horizontal flips + late decay", "300 / EMA")]
    final, identities = [], []
    for name, label, mode in names:
        record = read_json(inputs / "final_evaluation" / name / "metrics.json")
        metrics = record["metrics"][POOLED]
        if metrics["num_real_images"] != 4000 or metrics["num_generated_images"] != 4000:
            raise ValueError("Final comparison must use 4000 real and generated images")
        identities.append((record["prior_array_sha256"], record["reference_indices"],
                           metrics["protocol_hash"], record["plan_sha256"]))
        final.append({"model": label, "mode": mode, "fid": metrics["fid"], "is": metrics["inception_score"]})
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("Final models must share references and latent draws")
    table(tables / "final.tex", ["Model", "Epoch and weights", r"Test FID $\downarrow$", r"Test IS $\uparrow$"],
          [[r["model"], r["mode"], f'{r["fid"]:.2f}', f'{r["is"]:.2f}'] for r in final], "llrr")

    metadata = [read_json(inputs / "generation" / name / "generation.json") for name, _, _ in names]
    latent_hashes = [record.get("latent_array_sha256", record.get("latent_vectors_sha256")) for record in metadata]
    if len(set(latent_hashes)) != 1 or latent_hashes[0] is None:
        # Historical metadata calls this field latents_sha256.
        latent_hashes = [record["latents_sha256"] for record in metadata]
        if len(set(latent_hashes)) != 1:
            raise ValueError("Comparison grid requires shared latent vectors")
    fig, axes = plt.subplots(4, 8, figsize=(7, 2.9))
    labels = ["Real paintings\n(eight styles)", "Original\nepoch 95", "Unaugmented\nepoch 250, EMA", "Flips + decay\nepoch 300, EMA"]
    for row, name in enumerate(["real", "original95", "matched_C0F", "winner"]):
        for col in range(8):
            pixels = np.asarray(Image.open(inputs / "generation" / name / f"sample_{col:05d}.png"))
            if pixels.shape != (64, 64, 3) or pixels.dtype != np.uint8:
                raise ValueError("Display only the retained native RGB64 pixels")
            axes[row, col].imshow(pixels, interpolation="nearest")
            axes[row, col].axis("off")
        axes[row, 0].text(-.1, .5, labels[row], transform=axes[row, 0].transAxes,
                          ha="right", va="center", fontsize=8)
    fig.subplots_adjust(left=.17, right=.998, top=.995, bottom=.005, wspace=.025, hspace=.035)
    fig.savefig(figures / "generated_comparison.png", dpi=240)
    plt.close(fig)

    numbers = {"original_selection": selection["arms"],
               "historical_test": {k: original_test[k] for k in ("fid", "inception_score", "num_generated_images")},
               "validation_progression": trajectories, "matched_validation": refinement,
               "final_test": final,
               "fid_reduction_percent": 100 * (final[0]["fid"] - final[2]["fid"]) / final[0]["fid"],
               "is_increase_percent": 100 * (final[2]["is"] - final[0]["is"]) / final[0]["is"],
               "grid_latent_sha256": latent_hashes[0]}
    (output / "reported_numbers.json").write_text(json.dumps(numbers, indent=2) + "\n")
    print(json.dumps({"figures": 3, "tables": 3, "output": str(output),
                      "final_test": final}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "report")
    args = parser.parse_args()
    build(args.repository, args.output_dir)
