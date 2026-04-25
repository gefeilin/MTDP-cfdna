import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
DEATH_ONLY_CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, "death_only")

DEFAULT_AE_CHECKPOINT_PATH = os.path.join(
    DEATH_ONLY_CHECKPOINT_DIR,
    "autoencoder_v1_4_cfdna_torch_death_only.pt",
)
DEFAULT_BEST_MODEL_CHECKPOINT_PATH = os.path.join(
    DEATH_ONLY_CHECKPOINT_DIR,
    "best_model_v1_4_cfdna_torch_death_only.pt",
)


def ensure_checkpoint_dirs() -> None:
    os.makedirs(DEATH_ONLY_CHECKPOINT_DIR, exist_ok=True)


def sanitize_study_name(study_name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in study_name)


def build_optuna_fold_checkpoint_paths(
    save_dir: str,
    study_name: str,
    trial_number: int,
    fold_idx: int,
) -> tuple[str, str]:
    checkpoint_dir = os.path.join(save_dir, "runtime_checkpoints", "death_only")
    os.makedirs(checkpoint_dir, exist_ok=True)
    safe_study_name = sanitize_study_name(study_name)
    prefix = f"{safe_study_name}_trial{trial_number}_fold{fold_idx}"
    ae_ckpt_path = os.path.join(checkpoint_dir, f"{prefix}_autoencoder.pt")
    best_ckpt_path = os.path.join(checkpoint_dir, f"{prefix}_best_model.pt")
    return ae_ckpt_path, best_ckpt_path


def build_retrain_checkpoint_paths(
    save_dir: str,
    study_name: str,
    trial_number: int,
    tag: str,
) -> tuple[str, str]:
    checkpoint_dir = os.path.join(save_dir, "retrain_checkpoints", "death_only")
    os.makedirs(checkpoint_dir, exist_ok=True)
    safe_study_name = sanitize_study_name(study_name)
    safe_tag = sanitize_study_name(tag)
    prefix = f"{safe_study_name}_trial{trial_number}_{safe_tag}"
    ae_ckpt_path = os.path.join(checkpoint_dir, f"{prefix}_autoencoder.pt")
    best_ckpt_path = os.path.join(checkpoint_dir, f"{prefix}_best_model.pt")
    return ae_ckpt_path, best_ckpt_path
