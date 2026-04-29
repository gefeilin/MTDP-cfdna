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
- Reuse saved SHAP caches when they exist, then fall back to saved explainers. The app does not run live Kernel SHAP at request time.

Run:

```bash
cd /data/ling2/Sean_project/cfDNA_project/cfdna_multitask_project_organized/app
python app.py
```

Notes:
- The app intentionally does not depend on the old SCD app layout.
- It avoids the old conformal calibration flow and uses MC-dropout uncertainty instead.
- The app now loads fixed trial 506 FEV1 scaling metadata from `app/data/model_metadata.json`.
- The app now tries to load saved individual SHAP pickle caches from `results/kernel_shap_cache_trial506_noaucddpct` by default.
- The app can also load target-specific saved explainers from `app/data/saved_explainers`.
- Set `CFDNA_USE_SAVED_SHAP=0` to skip saved per-patient SHAP caches.
- If a saved SHAP explainer is missing, rebuild it offline with `scripts/build_saved_explainers.py` before running the app.
