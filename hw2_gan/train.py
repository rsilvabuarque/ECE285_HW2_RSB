"""Train DCGAN; select checkpoints by fixed validation probability-FID."""

import argparse
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from .data import CachedPaintings
from .evaluate import evaluate_generator
from .model import Generator, Discriminator, initialize_weights, adversarial_step
from .utils import (require_compute, seed_everything, resolve_device, save_json,
                    save_grid, save_checkpoint, record_provenance)


def plot_history(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
    for axis,key,label in zip(axes,["g_loss","d_loss","val_fid"],
                             ["Generator BCE","Discriminator BCE (sum)","Validation probability-FID"]):
        present=[row for row in rows if row.get(key) is not None]
        axis.plot([row["epoch"] for row in present],[row[key] for row in present])
        axis.set(xlabel="Epoch",ylabel=label);axis.grid(alpha=.2)
    fig.tight_layout();fig.savefig(path,dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--epochs",type=int,default=50)
    parser.add_argument("--latent-dim",type=int,default=100)
    parser.add_argument("--base-channels",type=int,default=64)
    parser.add_argument("--batch-size",type=int,default=128)
    parser.add_argument("--learning-rate",type=float,default=.0002)
    parser.add_argument("--beta1",type=float,default=.5)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--num-workers",type=int,default=4)
    parser.add_argument("--eval-every",type=int,default=5)
    parser.add_argument("--val-samples",type=int,default=1024)
    parser.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    parser.add_argument("--resume",type=Path)
    args=parser.parse_args()
    require_compute();torch.set_num_threads(2)
    if args.epochs<1 or args.batch_size<2 or args.eval_every<1:
        raise ValueError("Invalid epochs, batch size, or evaluation interval.")
    state=torch.load(args.resume,map_location="cpu",weights_only=True) if args.resume else None
    config={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}
    if state:
        for key in ["latent_dim","base_channels","batch_size","learning_rate","beta1","seed",
                    "num_workers","eval_every","val_samples"]:
            config[key]=state["config"][key]
        if args.epochs<=state["epoch"]: raise ValueError("Requested epochs already completed.")
    output=args.output_dir
    if not state and any((output/name).exists() for name in ["last.pt","best.pt","history.csv"]):
        raise FileExistsError("Use a new run directory or resume its last.pt.")
    if state and args.resume.parent.resolve()!=output.resolve():
        raise ValueError("Resume in the original run directory to preserve best.pt.")
    output.mkdir(parents=True,exist_ok=True)
    device=resolve_device(args.device);seed_everything(config["seed"])
    train=CachedPaintings(args.cache_dir,"train")
    fingerprint=train.manifest["fingerprint"]
    if state and state["cache_fingerprint"]!=fingerprint:
        raise ValueError("Dataset cache differs from the saved run.")
    data_rng=torch.Generator().manual_seed(config["seed"])
    noise_rng=torch.Generator(device=device).manual_seed(config["seed"]+2)
    loader=DataLoader(train,batch_size=config["batch_size"],shuffle=True,num_workers=config["num_workers"],
                      drop_last=False,pin_memory=device.type=="cuda",generator=data_rng,
                      persistent_workers=config["num_workers"]>0)
    G=Generator(config["latent_dim"],config["base_channels"]).to(device)
    D=Discriminator(config["base_channels"]).to(device)
    G.apply(initialize_weights);D.apply(initialize_weights)
    g_optimizer=torch.optim.Adam(G.parameters(),lr=config["learning_rate"],betas=(config["beta1"],.999))
    d_optimizer=torch.optim.Adam(D.parameters(),lr=config["learning_rate"],betas=(config["beta1"],.999))
    history=[];best_val=math.inf;best_epoch=None;start_epoch=1
    if state:
        G.load_state_dict(state["generator_state"]);D.load_state_dict(state["discriminator_state"])
        g_optimizer.load_state_dict(state["g_optimizer_state"]);d_optimizer.load_state_dict(state["d_optimizer_state"])
        data_rng.set_state(state["data_rng_state"]);noise_rng.set_state(state["noise_rng_state"])
        torch.set_rng_state(state["torch_rng_state"])
        if device.type=="cuda": torch.cuda.set_rng_state_all(state["cuda_rng_states"])
        history=state["history"];best_val=state["best_val_fid"];best_epoch=state["best_epoch"]
        start_epoch=state["epoch"]+1
        best_path=output/"best.pt"
        if best_epoch==state["epoch"]:
            selected={key:state[key] for key in ["generator_state","discriminator_state","epoch",
                      "config","history","cache_fingerprint","best_val_fid","best_epoch"]}
            if not best_path.exists() or torch.load(best_path,map_location="cpu",weights_only=True)["epoch"]!=best_epoch:
                save_checkpoint(best_path,selected)
        elif best_epoch is not None and not best_path.is_file():
            raise FileNotFoundError("Keep the selected best.pt beside last.pt when resuming.")
    save_json(output/"config.json",config)
    save_json(output/"dataset_summary.json",{key:train.manifest[key] for key in
              ["dataset","dataset_version","fingerprint","preprocessing","split_rule","split_seed","style_counts","failures","limitations"]} |
              {"split_counts":{key:len(value) for key,value in train.manifest["splits"].items()}})
    record_provenance(output,config)
    fixed_rng=torch.Generator().manual_seed(43)
    fixed=torch.randn(64,G.latent_dim,generator=fixed_rng).to(device)
    print("Parameters:",sum(p.numel() for p in G.parameters()),"G;",sum(p.numel() for p in D.parameters()),"D",flush=True)
    for epoch in range(start_epoch,args.epochs+1):
        G.train();D.train();start=time.monotonic();sums={};seen=0
        for batch,real in enumerate(loader,1):
            real=real.to(device,non_blocking=True)
            metrics=adversarial_step(G,D,real,g_optimizer,d_optimizer,noise_rng)
            for key,value in metrics.items(): sums[key]=sums.get(key,0)+len(real)*value
            seen+=len(real)
            if batch%200==0:
                print(f"epoch {epoch} batch {batch}/{len(loader)} G={metrics['g_loss']:.4f} D={metrics['d_loss']:.4f}",flush=True)
        if device.type=="cuda": torch.cuda.synchronize()
        train_seconds=time.monotonic()-start
        row={"epoch":epoch,"seconds":train_seconds,**{key:value/seen for key,value in sums.items()},
             "val_fid":None,"val_is":None,"validation_seconds":0.0}
        selected=False
        if epoch==2 or epoch%config["eval_every"]==0 or epoch==args.epochs:
            evaluation_start=time.monotonic()
            stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            rng_before=torch.get_rng_state().clone()
            cuda_before=torch.cuda.get_rng_state(device).clone() if device.type=="cuda" else None
            buffers_before={key:value.detach().clone() for key,value in G.named_buffers()}
            validation=evaluate_generator(G,args.cache_dir,"val",config["val_samples"],43,device,
                                          output/"validation"/f"epoch_{epoch:03d}_{stamp}")
            assert torch.equal(rng_before,torch.get_rng_state()),"Validation advanced training CPU RNG."
            if cuda_before is not None: assert torch.equal(cuda_before,torch.cuda.get_rng_state(device))
            assert G.training and all(torch.equal(value,buffers_before[key]) for key,value in G.named_buffers())
            row.update(val_fid=validation["fid"],val_is=validation["inception_score"],
                       validation_seconds=time.monotonic()-evaluation_start)
            if row["val_fid"]<best_val:
                best_val=row["val_fid"];best_epoch=epoch;selected=True
        history.append(row)
        temporary=output/"history.csv.tmp"
        with temporary.open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(row));writer.writeheader();writer.writerows(history)
        temporary.replace(output/"history.csv")
        common={"generator_state":G.state_dict(),"discriminator_state":D.state_dict(),"epoch":epoch,
                "config":config,"history":history,"cache_fingerprint":fingerprint,
                "best_val_fid":best_val,"best_epoch":best_epoch}
        last=common | {"g_optimizer_state":g_optimizer.state_dict(),"d_optimizer_state":d_optimizer.state_dict(),
                       "data_rng_state":data_rng.get_state(),"noise_rng_state":noise_rng.get_state(),
                       "torch_rng_state":torch.get_rng_state(),
                       "cuda_rng_states":torch.cuda.get_rng_state_all() if device.type=="cuda" else None}
        save_checkpoint(output/"last.pt",last)
        if selected:
            save_checkpoint(output/"best.pt",common)
        if row["val_fid"] is not None:
            save_checkpoint(output/"checkpoints"/f"epoch_{epoch:03d}.pt",common)
        G.eval()
        with torch.inference_mode(): save_grid(G(fixed),output/"previews"/f"epoch_{epoch:03d}.png")
        G.train()
        plot_history(history,output/"loss_curves.png")
        print(f"Epoch {epoch}/{args.epochs}: G={row['g_loss']:.4f} D={row['d_loss']:.4f}, "
              f"valFID={row['val_fid']}, train={train_seconds:.1f}s, best={best_epoch}",flush=True)
    print("Completed",len(history),"epochs; selected",best_epoch,"validation FID",best_val,flush=True)


if __name__=="__main__": main()
