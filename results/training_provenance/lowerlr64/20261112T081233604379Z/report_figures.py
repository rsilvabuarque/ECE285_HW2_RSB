"""Build report figures from measured histories and saved generated paintings."""

import argparse
import csv
import json
from pathlib import Path
import shutil

from PIL import Image


def read_history(path):
    with Path(path).open(newline="") as handle:
        return [{key:(int(value) if key=="epoch" else float(value)) if value else None
                 for key,value in row.items()} for row in csv.DictReader(handle)]


def plot_training(rows, selected_epoch, output):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    plt.rcParams.update({"font.size":8,"axes.labelsize":8,"legend.fontsize":7})
    fig,axes=plt.subplots(1,3,figsize=(6.4,1.9))
    for axis,key,label,color in zip(axes,["g_loss","d_loss","val_fid"],
           ["Generator BCE","Discriminator BCE (sum)","Validation probability-FID"],
           ["#2266aa","#cc7733","#228833"]):
        measured=[row for row in rows if row.get(key) is not None]
        axis.plot([row["epoch"] for row in measured],[row[key] for row in measured],
                  color=color,linewidth=1.2,marker="o" if key=="val_fid" else None,markersize=2)
        axis.axvline(selected_epoch,color=".35",linestyle="--",linewidth=.8)
        axis.set(xlabel="Epoch",ylabel=label);axis.grid(alpha=.2)
    fig.tight_layout(pad=.5)
    for suffix in ["pdf","png"]:fig.savefig(output/f"training_diagnostics.{suffix}",dpi=240)
    plt.close(fig)


def tile_samples(directory, output):
    panel=Image.new("RGB",(266,134))
    for index in range(8):
        with Image.open(directory/f"sample_{index:05d}.png") as image:
            if image.size!=(64,64):raise ValueError("Expected unmodified RGB64 sample tiles.")
            panel.paste(image.convert("RGB"),(2+(index%4)*66,2+(index//4)*66))
    panel.save(output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--samples-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    selection=json.loads((args.run_dir/"selection.json").read_text())
    generation=json.loads((args.samples_dir/"generation.json").read_text())
    if generation["checkpoint_sha256"]!=selection["checkpoint_sha256"]:
        raise ValueError("Generated examples do not use the frozen checkpoint.")
    if args.output_dir.exists():raise FileExistsError("Use a new output directory.")
    args.output_dir.mkdir(parents=True)
    rows=read_history(args.run_dir/"history.csv")
    plot_training(rows,selection["selected_epoch"],args.output_dir)
    tile_samples(args.samples_dir,args.output_dir/"prior_samples_report.png")
    for name in ["generation.json","samples_first16.png","samples_grid.png"]:
        source=args.samples_dir/name
        if source.exists():shutil.copy2(source,args.output_dir/name)
    (args.output_dir/"figure_inputs.json").write_text(json.dumps({
        "selection":selection,"generation":generation,"history_epochs":len(rows),
        "tiles":"first eight saved PNGs, four columns, two-pixel black border; no pixel rescaling"},indent=2)+'\n')
    print("Saved report figures to",args.output_dir)


if __name__=="__main__":main()
