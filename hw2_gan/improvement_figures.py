"""Regenerate report figures from retained pixels and measured CSV histories.

No raw WikiArt archive, model inference or training is needed. Display always
uses the stored uint8 RGB pixels; magnification uses nearest-neighbor rendering.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image


def build(results,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    results,output=Path(results),Path(output);output.mkdir(parents=True,exist_ok=True)
    config=json.loads((results/"figure_config.json").read_text())
    fig,axs=plt.subplots(4,8,figsize=(7,3.1))
    labels=["Real training\n(8 styles)"]
    for name,label in [("original95","Original"),("matched_C0F","Matched C0F"),("winner",config["winner_family"])]:
        spec=config["models"][name]
        labels.append(f"{label}\ne{spec['epoch']}/{spec['variant'].upper()}\nseed {spec['seed']}")
    for row,name in enumerate(["real","original95","matched_C0F","winner"]):
        for col in range(8):
            pixels=np.asarray(Image.open(results/"generation"/name/f"sample_{col:05d}.png"))
            if pixels.shape!=(64,64,3):raise ValueError("Core comparison requires native RGB64 tiles.")
            axs[row,col].imshow(pixels,interpolation="nearest");axs[row,col].axis("off")
        axs[row,0].text(-.08,.5,labels[row],transform=axs[row,0].transAxes,ha="right",va="center",fontsize=8)
    fig.subplots_adjust(left=.16,right=.998,top=.99,bottom=.01,wspace=.025,hspace=.04)
    fig.savefig(output/"fixed_comparison.png",dpi=220);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.8,1.5))
    for family,color in dict([("C0F","tab:gray"),(config["winner_family"],"tab:blue")]).items():
        with (results/"histories"/f"{family}_s42.csv").open() as f:history=list(csv.DictReader(f))
        for variant,field,ls in [("raw","val_pool_fid","--"),("EMA","val_ema_pool_fid","-")]:
            points=[(int(r["epoch"]),float(r[field])) for r in history if r.get(field)]
            ax.plot(*zip(*points),ls+"o",color=color,label=f"{family} {variant}",ms=3,lw=1)
    ax.set(xlabel="Epoch (seed42 discovery)",ylabel="Validation pooled FID");ax.grid(alpha=.2);ax.legend(ncol=4,fontsize=7,loc="upper right")
    fig.tight_layout(pad=.5);fig.savefig(output/"screening_trajectory.png",dpi=200);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(6.8,2.2))
    for family,color in dict([("C0F","tab:gray"),(config["winner_family"],"tab:blue")]).items():
        with (results/"histories"/f"{family}_s42.csv").open() as f:history=list(csv.DictReader(f))
        for ax,field,label in zip(axes,["g_loss","d_loss"],["G non-saturating BCE","D objective (includes R1 where used)"]):
            ax.plot([int(r["epoch"]) for r in history],[float(r[field]) for r in history],label=family,color=color,lw=.8)
            ax.set(xlabel="Epoch (seed42)",ylabel=label);ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout(pad=.7);fig.savefig(output/"loss_trajectory.png",dpi=200);plt.close(fig)
    # One compact report panel keeps both objectives and the actual selection
    # metric visible. These are discovery histories, not representative-seed runs.
    fig,axes=plt.subplots(1,3,figsize=(6.8,1.7))
    for family,color in dict([("C0F","tab:gray"),(config["winner_family"],"tab:blue")]).items():
        with (results/"histories"/f"{family}_s42.csv").open() as f:history=list(csv.DictReader(f))
        epochs=[int(r["epoch"]) for r in history]
        for ax,field in zip(axes[:2],["g_loss","d_loss"]):
            ax.plot(epochs,[float(r[field]) for r in history],label=family,color=color,lw=.8)
        for field,style,label in [("val_pool_fid","--o","raw"),("val_ema_pool_fid","-o","EMA")]:
            points=[(int(r["epoch"]),float(r[field])) for r in history if r.get(field)]
            axes[2].plot(*zip(*points),style,color=color,ms=2,lw=.8,label=f"{family} {label}")
    for ax,label in zip(axes,["G BCE","D objective","Validation pooled FID"]):
        ax.set(xlabel="Epoch (seed42)",ylabel=label);ax.grid(alpha=.2);ax.tick_params(labelsize=7)
        ax.xaxis.label.set_size(8);ax.yaxis.label.set_size(8);ax.legend(fontsize=6,loc="best")
    fig.tight_layout(pad=.45,w_pad=.6);fig.savefig(output/"training_diagnostics.png",dpi=240);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument("--results-dir",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);a=p.parse_args();build(a.results_dir,a.output_dir)


if __name__=="__main__":main()
