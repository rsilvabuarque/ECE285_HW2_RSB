# Frozen measurements used by the final report

`improvement/` copies measured JSON/CSV files and the first eight image tiles
from each common-latent comparison row, from the supplementary release at commit
`e4e0d5604abd67d3aae4617a7e7449d8673de768`. `SOURCE_FILES.json` records their
SHA256 hashes. These are observations, not fresh training outputs. Historical
absolute paths inside provenance JSONs identify the source run; rebuilding the
report does not follow those paths.

The original campaign inputs remain in `results/` at the repository root: selected and lower-rate histories/configurations,
selection, dataset summary/manifest and historical test metrics. The original
selection JSON binds both histories and configurations by SHA256.

`all_epoch_metrics.csv` uses run and epoch as keys. `val_ema_pool_fid` is the
pooled-feature validation FID used in the refinement figure/table. `val_is` and
`val_ema_is` are historical 1024-image monitoring IS and are not used as the
4000-image final IS. The final table reads each model's complete
`final_evaluation/*/metrics.json`, selecting `torchvision2048pool-v1`.

Curves show seeds123 and314 separately as thin lines and their arithmetic mean
as a thick line. Constant and decaying extensions share the same epoch200
parents; repeated parent rows in the combined CSV are not extra training data.
The real image row illustrates eight uniformly sampled styles, one training
image each, rather than the dataset's style proportions. Generated rows use
indices0–7 from64 common standard-normal latent draws at seed43, with fixed
intensity conversion. Generation metadata and real-image provenance are retained.

Run `python scripts/rebuild_report.py` from the repository root to verify the
input hashes and recreate all figures, tables and numerical summaries offline.
See `additional_analysis` for complete experiment code, other configurations,
selected model weights and full64-image grids.
