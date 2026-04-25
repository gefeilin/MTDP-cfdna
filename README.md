cfDNA Multitask DeepHit App

This app is a fresh Dash frontend for the cfDNA multitask DeepHit trial 506 model.

What it is built around:
- Trial `506`
- Study `multitask_deephit_v1_4_cfdna_optuna_death_only_v5_maskhead_nosite_nopeakdsa_noweights_noaucddpct-20260418v1`
- Checkpoint resolved from the Optuna SQLite database
- SHAP cache directory `results/kernel_shap_cache_trial506_noaucddpct`

Main features:
- Upload cfDNA baseline predictor CSVs
- Predict 2-year mortality, 1-year FEV1, and three binary intermediate outcomes
- Show MC-dropout uncertainty bands for survival and FEV1
- Edit a selected patient row and rerun the analysis
- Reuse saved SHAP caches when they exist, then fall back to saved explainers, then to live Kernel SHAP
- Ship with the trial 506 checkpoint, cohort schema files, Optuna DB, and saved SHAP cache needed for standalone deployment

Run:

```bash
cd MTDP-cfdna
python app.py
```

Notes:
- The app intentionally does not depend on the old SCD app layout.
- It avoids the old conformal calibration flow and uses MC-dropout uncertainty instead.
- The app loads fixed trial 506 FEV1 scaling metadata from `data/model_metadata.json`.
- The app also loads the fixed trial 506 network structure and B-spline settings from `data/model_metadata.json`, so Optuna DB metadata is no longer needed for inference.
- Full training normalization and categorical encoding metadata are frozen in `data/schema_metadata.json`.
- The app tries to load saved individual SHAP pickle caches from `results/kernel_shap_cache_trial506_noaucddpct` by default.
- The app can also load target-specific saved explainers from `data/saved_explainers`.
- The repository now includes the trial 506 runtime assets required for standalone hosting:
  - `data/demo_survival_supplementary_no30_death_only_v5.csv` containing only the first 10 sample rows
  - `data/schema_metadata.json` containing frozen full-training schema statistics
  - `results/death_only/.../trial506_all_data_retrain_best_model.pt`
  - `results/kernel_shap_cache_trial506_noaucddpct/*.pkl` trimmed to the minimal sample cache set
- Minimal engine files are vendored into `engines/` so the app can run outside the original cfDNA project tree.
- `app.py` supports `PORT` and `HOST` environment variables for hosted deployment targets such as Posit Cloud.
- Set `CFDNA_USE_SAVED_SHAP=0` to skip saved per-patient SHAP caches.
- Set `CFDNA_USE_SAVED_EXPLAINER=0` only if you want to force live Kernel SHAP instead of saved explainers.
