import os
import copy
import inspect
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset
from scipy.interpolate import BSpline
from sksurv.metrics import concordance_index_censored
try:
    from torchjd.autojac import mtl_backward as torchjd_mtl_backward
    from torchjd.aggregation import UPGrad as TorchJDUPGrad
except Exception:
    torchjd_mtl_backward = None
    TorchJDUPGrad = None


def _torchjd_mtl_backward_compat(
    objective_losses,
    shared_features,
    jd_aggregator,
    tasks_params,
    shared_params,
):
    if torchjd_mtl_backward is None:
        raise RuntimeError("torchjd is not available")

    signature = inspect.signature(torchjd_mtl_backward)
    supported = signature.parameters
    kwargs = {}

    if "features" in supported:
        kwargs["features"] = shared_features
    if "aggregator" in supported:
        kwargs["aggregator"] = jd_aggregator
    elif "aggregation" in supported:
        kwargs["aggregation"] = jd_aggregator
    if "tasks_params" in supported:
        kwargs["tasks_params"] = tasks_params
    if "shared_params" in supported:
        kwargs["shared_params"] = shared_params
    if "retain_graph" in supported:
        kwargs["retain_graph"] = True
    if "parallel_chunk_size" in supported:
        kwargs["parallel_chunk_size"] = 1

    return torchjd_mtl_backward(objective_losses, **kwargs)

TIME_BIN_EDGES_YEARS = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
TIME_BIN_REPRESENTATIVES_YEARS = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 5.5], dtype=np.float32)
# Event-specific IBS horizons in years (coarse values close to median days).
EVENT_IBS_HORIZON_YEARS = {0: 2.0, 1: 2.0}
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SAVE_DIR = os.path.join(SCRIPT_DIR, "saved_model")
DEFAULT_AE_SAVE_PATH = os.path.join(DEFAULT_SAVE_DIR, "autoencoder_v1.2.2_cfdna_torch_death_only.pt")
DEFAULT_BEST_MODEL_PATH = os.path.join(
    DEFAULT_SAVE_DIR,
    "best_model_training_multi_long_v1.2.2_cfdna_torch_death_only.pt",
)


def log(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x + 1e-8)


def get_activation_fn(name: str):
    activations = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "elu": nn.ELU(),
        "softmax": nn.Softmax(dim=-1),
    }
    return activations.get(name, None)


def overall_cause_specific_c_index(pred, event, time, num_causes_idx):
    out = pred[:, num_causes_idx, :].detach().cpu().numpy()
    time_points = get_time_bin_representatives_years(out.shape[1])
    expected_time = np.sum(out * time_points, axis=1)
    risk_score = -expected_time
    event_np = event.detach().cpu().numpy().squeeze()
    # Cause-specific C-index: only the target cause is treated as an event.
    # Censoring (0) and competing causes are treated as censored.
    event_indicator_bool = (event_np == (num_causes_idx + 1))
    try:
        c_index = concordance_index_censored(
            event_indicator_bool.squeeze(),
            time.detach().cpu().numpy().squeeze(),
            risk_score.squeeze(),
        )[0]
    except Exception:
        c_index = -1
    return c_index


def get_time_bin_representatives_years(n_bins: int) -> np.ndarray:
    """
    Representative time (years) for each custom bin:
    [<=3m, <=6m, <=9m, <=1y, <=18m, <=2y, <=3y, <=4y, <=5y, >5y]
    """
    if n_bins <= TIME_BIN_REPRESENTATIVES_YEARS.shape[0]:
        return TIME_BIN_REPRESENTATIVES_YEARS[:n_bins]
    # Fallback for unexpected larger bin count
    return np.arange(1, n_bins + 1, dtype=np.float32)


def build_ibs_grid_from_horizon_years(horizon_years=None):
    """
    Build IBS integration grids based on discrete time bins.
    Returns:
      bin_grid:  [0, 1, ..., max_bin]
      year_grid: [0, rep_1, ..., rep_max_bin]
    """
    n_bins = TIME_BIN_REPRESENTATIVES_YEARS.shape[0]
    if horizon_years is None:
        max_bin = n_bins
    else:
        max_bin = int(np.searchsorted(TIME_BIN_EDGES_YEARS, float(horizon_years), side="left") + 1)
        max_bin = max(1, min(max_bin, n_bins))

    bin_grid = np.arange(0, max_bin + 1, dtype=np.int64)
    year_grid = np.concatenate(([0.0], TIME_BIN_REPRESENTATIVES_YEARS[:max_bin])).astype(np.float32)
    return bin_grid, year_grid


def cause_specific_intergrated_brier_score(predictions, time_survival, event_type, num_causes_idx):
    prediction = predictions[:, num_causes_idx, :].detach().cpu().numpy()
    event_type = event_type.detach().cpu().numpy().squeeze()
    time_survival = time_survival.detach().cpu().numpy().squeeze()
    bin_grid, year_grid = build_ibs_grid_from_horizon_years(
        EVENT_IBS_HORIZON_YEARS.get(num_causes_idx, None)
    )
    cause_label = num_causes_idx + 1

    brier_scores = []
    for b in bin_grid:
        if b == 0:
            pred_e = np.zeros(prediction.shape[0])
        else:
            pred_e = np.sum(prediction[:, : int(b)], axis=1)

        y_true = ((time_survival <= b) & (event_type == cause_label)).astype(float)
        brier_scores.append(np.mean((pred_e - y_true) ** 2))

    trapz_fn = getattr(np, "trapezoid", None)
    if trapz_fn is None:
        trapz_fn = np.trapz
    denom = float(year_grid[-1] - year_grid[0]) if len(year_grid) > 1 else 1.0
    ibs = trapz_fn(brier_scores, year_grid) / max(denom, 1e-8)
    return brier_scores, ibs


def f_get_Normalization(X, norm_mode):
    X = np.array(X, dtype=np.float32, copy=True)
    num_patient, num_feature = np.shape(X)

    if norm_mode == "standard":
        for j in range(num_feature):
            if np.std(X[:, j]) != 0:
                X[:, j] = (X[:, j] - np.mean(X[:, j])) / np.std(X[:, j])
            else:
                X[:, j] = X[:, j] - np.mean(X[:, j])
    elif norm_mode == "normal":
        for j in range(num_feature):
            X[:, j] = (X[:, j] - np.min(X[:, j])) / (np.max(X[:, j]) - np.min(X[:, j]))
    else:
        raise ValueError("INPUT MODE ERROR")

    return X


def f_get_fc_mask2(time, label, num_Event, num_Category):
    mask = np.zeros([np.shape(time)[0], num_Event, num_Category])
    for i in range(np.shape(time)[0]):
        if label[i, 0] != 0:
            time_idx = min(int(time[i, 0] - 1), num_Category - 1)
            mask[i, int(label[i, 0] - 1), time_idx] = 1
        else:
            time_idx = min(int(time[i, 0] - 1), num_Category)
            mask[i, :, time_idx:] = 1
    return mask


def f_get_fc_mask3(time, meas_time, num_Category):
    mask = np.zeros([np.shape(time)[0], num_Category])
    if np.shape(meas_time):
        for i in range(np.shape(time)[0]):
            t1 = int(meas_time[i, 0])
            t2 = int(time[i, 0])
            mask[i, (t1 + 1) : (t2 + 1)] = 1
    else:
        for i in range(np.shape(time)[0]):
            t = int(time[i, 0])
            mask[i, :t] = 1
    return mask


def factorize_df(df: pd.DataFrame) -> np.ndarray:
    """Factorize each column to int codes, return float32 matrix."""
    out = df.copy()
    for c in out.columns:
        out[c] = pd.factorize(out[c], sort=True)[0]  # -1 for NaN
    return out.to_numpy(dtype=np.float32)


def missing_mask_from_df(df: pd.DataFrame) -> np.ndarray:
    """Boolean mask: True means observed."""
    return (~df.isna()).to_numpy(dtype=bool)


def discretize_time_to_custom_bins(time_days: np.ndarray) -> np.ndarray:
    """
    Discretize event time (days) into 10 bins:
    1) <=3m, 2) <=6m, 3) <=9m, 4) <=1y, 5) <=18m,
    6) <=2y, 7) <=3y, 8) <=4y, 9) <=5y, 10) >5y.
    """
    edges = TIME_BIN_EDGES_YEARS * 365.25
    # bin indices in [1..10]
    bins = np.searchsorted(edges, time_days.astype(np.float32), side="left") + 1
    return bins.astype(np.float32)


def import_dataset_cfdna_sim_outcome(norm_mode="standard"):
    df_with_miss = pd.read_pickle(
        "/data/ling2/Sean_project/cfDNA_project/data/demo_survival_supplementary_no30_pft_io_death_only.pkl"
    )
    # Keep only subjects with event_time > 30 days.
    df_with_miss = df_with_miss[df_with_miss["event_time"] > 30].copy()
    df_baseline_with_miss = df_with_miss.drop(
        columns=["FVC_PCT_matrix", "FEV1_PCT_matrix", "severe_ACR", "ever_clinical_AMR", "BLAD"]
    ).copy()
    df_long_for_pred = df_with_miss[["FVC_PCT_matrix", "FEV1_PCT_matrix"]].copy()
    df_baseline_imputed = pd.read_csv(
        "/data/ling2/Sean_project/cfDNA_project/data/demo_survival_supplementary_no30_death_only.csv"
    )
    df_baseline_imputed = df_baseline_imputed[df_baseline_imputed["event_time"] > 30].copy()
    # Align rows across baseline/imputed and longitudinal data by SUBJECT_NUMBER.
    common_ids = np.intersect1d(
        df_baseline_imputed["SUBJECT_NUMBER"].to_numpy(),
        df_with_miss["SUBJECT_NUMBER"].to_numpy(),
    )
    df_baseline_imputed = (
        df_baseline_imputed[df_baseline_imputed["SUBJECT_NUMBER"].isin(common_ids)]
        .sort_values("SUBJECT_NUMBER")
        .reset_index(drop=True)
    )
    df_with_miss = (
        df_with_miss[df_with_miss["SUBJECT_NUMBER"].isin(common_ids)]
        .sort_values("SUBJECT_NUMBER")
        .reset_index(drop=True)
    )
    df_baseline_with_miss = df_with_miss.drop(
        columns=["FVC_PCT_matrix", "FEV1_PCT_matrix", "severe_ACR", "ever_clinical_AMR", "BLAD"]
    ).copy()
    df_long_for_pred = df_with_miss[["FVC_PCT_matrix", "FEV1_PCT_matrix"]].copy()

    cat_cols = [
        "final_TRANSPLANT_TYPE",
        "SITE",
        "SEX",
        "RACE",
        "ETHNICITY",
        "NATIVE_LUNG_DISEASE_Coded",
        "DONOR_CMV_SEROLOGY",
        "DONOR_GENDER",
        "DONOR_RACE",
        "DONOR_ETHNICITY",
        "DONOR_CAUSE_OF_DEATH",
        "DONOR_HX_DIABETES",
        "DONOR_HX_CANCER",
        "DONOR_HX_HYPERTENSION",
        "DONOR_HX_CIG",
        "DONOR_HX_ALCOHOL",
        "DONOR_HX_DRUG",
        "RECIPIENT_CMV_SEROLOGY",
        "DIABETES",
        "VENOUS_THROMBOEMBOLISM",
        "PULM_EMBOLISM",
        "SUPPLEMENTAL_OXYGEN",
        "VENTILATOR",
        "ECMO",
        "PGD_Grade_3",
        "DSA_present",
        "Peak_DSA_strength",
        "AMR_present",
        "HLA_MISMATCH_LVL",
        "BLOOD_TYPE_RECIPIENT",
        "BLOOD_TYPE_DONOR",
        "SMOKING_HISTORY_RECIPIENT",
    ]

    num_drop = ["SUBJECT_NUMBER", "event_time", "event"] + cat_cols
    df_numerical = df_baseline_imputed.drop(columns=num_drop)
    df_numerical_scaled = f_get_Normalization(df_numerical.to_numpy(dtype=np.float32), norm_mode)

    df_numerical_mean = np.array(df_numerical.mean()).reshape(-1, 1)
    df_numerical_std = np.array(df_numerical.std()).reshape(-1, 1)
    df_numerical_mean_std = np.concatenate((df_numerical_mean, df_numerical_std), axis=1)

    df_categorical = df_baseline_imputed[cat_cols].copy()
    X_cat = factorize_df(df_categorical)
    data = np.concatenate((df_numerical_scaled.astype(np.float32), X_cat), axis=1).astype(np.float32)

    label = np.asarray(df_baseline_imputed[["event"]])
    time_days = np.asarray(df_baseline_imputed[["event_time"]], dtype=np.float32)
    time = discretize_time_to_custom_bins(time_days)

    # Fixed 10 bins to stay aligned with predefined custom time bins.
    num_Category = int(TIME_BIN_REPRESENTATIVES_YEARS.shape[0])
    num_Event = int(len(np.unique(label)) - 1)

    mask1 = f_get_fc_mask2(time, label, num_Event, num_Category)
    mask2 = f_get_fc_mask3(time, -1, num_Category)

    miss_cat_cols = [
        "final_TRANSPLANT_TYPE",
        "SITE",
        "SEX",
        "RACE",
        "ETHNICITY",
        "NATIVE_LUNG_DISEASE_Coded",
        "DONOR_CMV_SEROLOGY",
        "DONOR_GENDER",
        "DONOR_RACE",
        "DONOR_ETHNICITY",
        "DONOR_CAUSE_OF_DEATH",
        "DONOR_HX_DIABETES",
        "DONOR_HX_CANCER",
        "DONOR_HX_HYPERTENSION",
        "DONOR_HX_CIG",
        "DONOR_HX_ALCOHOL",
        "DONOR_HX_DRUG",
        "RECIPIENT_CMV_SEROLOGY",
        "DIABETES",
        "VENOUS_THROMBOEMBOLISM",
        "PULM_EMBOLISM",
        "SUPPLEMENTAL_OXYGEN",
        "VENTILATOR",
        "ECMO",
        "PGD_Grade_3",
        "DSA_present",
        "Peak_DSA_strength",
        "AMR_present",
        "HLA_MISMATCH_LVL",
        "BLOOD_TYPE_RECIPIENT",
        "BLOOD_TYPE_DONOR",
        "SMOKING_HISTORY_RECIPIENT",
    ]
    miss_num_drop = ["SUBJECT_NUMBER", "event_time", "event"] + miss_cat_cols
    df_numerical_miss = df_baseline_with_miss.drop(columns=miss_num_drop)
    df_categorical_miss = df_baseline_with_miss[miss_cat_cols].copy()

    df_miss_all = pd.concat([df_numerical_miss, df_categorical_miss], axis=1)
    mask_miss = missing_mask_from_df(df_miss_all).astype(np.float32)

    outcome_matrix_pred_long = np.array(
        df_long_for_pred.drop(columns=["SUBJECT_NUMBER"], errors="ignore").to_numpy().tolist(),
        dtype=np.float64,
    )
    n_longitudinal = outcome_matrix_pred_long.shape[1]

    # Merge baseline binary outcomes into outcome matrix as extra outcome channels.
    # They only use the first time-point value (index [0, 0]) and keep NaN elsewhere.
    binary_cols = ["severe_ACR", "ever_clinical_AMR", "BLAD"]
    binary_raw = df_with_miss[binary_cols].to_numpy(dtype=np.float64)  # (N, 3), can contain NaN
    n_samples, _, t_dim, _ = outcome_matrix_pred_long.shape
    binary_tensor = np.full((n_samples, len(binary_cols), t_dim, 2), np.nan, dtype=np.float64)
    binary_tensor[:, :, 0, 0] = binary_raw
    binary_tensor[:, :, 0, 1] = 0.0

    outcome_matrix_pred = np.concatenate([outcome_matrix_pred_long, binary_tensor], axis=1)
    mask_miss_pred = ~np.isnan(outcome_matrix_pred)

    outcome_matrix_scaled = outcome_matrix_pred.copy()
    long_mean = []
    long_std = []
    for i in range(outcome_matrix_pred.shape[1]):
        x = outcome_matrix_pred[:, i, :, 0]
        if i < n_longitudinal:
            mean = np.nanmean(x)
            std = np.nanstd(x)
            long_mean.append(mean)
            long_std.append(std)
            outcome_matrix_scaled[:, i, :, 0] = (x - mean) / (std + 1e-8)

    long_mean = np.array(long_mean).reshape(-1, 1)
    long_std = np.array(long_std).reshape(-1, 1)
    long_mean_std = np.concatenate((long_mean, long_std), axis=1)

    outcome_matrix_scaled = np.nan_to_num(outcome_matrix_scaled, nan=0.0)
    x_time = outcome_matrix_scaled[:, :, :, 1]

    batch_size, covariate_dim, _ = x_time.shape
    k = 2
    num_internal_knots = 1
    internal_knots = np.linspace(0, 12, num_internal_knots + 2)[1:-1]
    t_knots = np.concatenate((np.repeat(0, k + 1), internal_knots, np.repeat(12, k + 1)))

    num_basis = len(t_knots) - (k + 1)
    c = np.eye(num_basis)
    b_spline_basis = [BSpline(t_knots, c[i], k) for i in range(num_basis)]

    batch_basis = []
    for b in range(batch_size):
        x_temp = x_time[b]
        basis_values_covariates = []
        for j in range(covariate_dim):
            t_input = x_temp[j]
            basis_eval = np.stack([basis(t_input) for basis in b_spline_basis], axis=0)
            basis_values_covariates.append(basis_eval)
        basis_values_covariates = np.stack(basis_values_covariates, axis=0)
        batch_basis.append(basis_values_covariates)

    batch_basis = np.stack(batch_basis, axis=0)

    return (
        data.astype("float32"),
        label.astype("float32"),
        time.astype("float32"),
        mask1.astype("float32"),
        mask2.astype("float32"),
        num_Category,
        num_Event,
        mask_miss.astype("float32"),
        outcome_matrix_scaled.astype("float32"),
        mask_miss_pred.astype("float32"),
        batch_basis.astype("float32"),
        df_numerical_mean_std.astype("float32"),
        long_mean_std.astype("float32"),
    )


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super().__init__()
        if feature_dim % num_heads != 0:
            raise ValueError("feature_dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        self.output_proj = nn.Linear(feature_dim, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x, mask=None):
        # x: (batch, input_dim, feature_dim)
        batch_size, input_dim, feature_dim = x.shape
        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)

        q = q.view(batch_size, input_dim, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, input_dim, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, input_dim, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            # mask: (batch, input_dim)
            attn_mask = mask.unsqueeze(1).unsqueeze(2)  # (batch,1,1,input_dim)
            attn_mask = attn_mask.expand(-1, self.num_heads, input_dim, -1)
            attention_scores = torch.where(
                attn_mask > 0,
                attention_scores,
                torch.full_like(attention_scores, -1e9),
            )

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, v)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, input_dim, feature_dim)
        multi_head_output = self.output_proj(attention_output)
        return self.norm(multi_head_output)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, num_heads):
        super().__init__()
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
        )
        self.attention = MultiHeadSelfAttention(hidden_dim2, num_heads)
        self.attention_norm = nn.LayerNorm(hidden_dim2)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim2)

    def forward(self, x, mask=None):
        x = self.input_embedding(x)
        attn_out = self.attention(x, mask=mask)
        x = self.attention_norm(x + attn_out)
        ffn_out = self.feed_forward(attn_out)
        x = self.ffn_norm(x + ffn_out)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, output_dim, hidden_dim1, hidden_dim2):
        super().__init__()
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, output_dim),
        )

    def forward(self, x):
        return self.output_layer(x)


class DTA_AE(nn.Module):
    """PyTorch version of the original attention autoencoder."""

    def __init__(self, hidden_dim1, hidden_dim2, num_heads, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.encoder_layers = nn.ModuleList()
        self.encoder_layers.append(
            TransformerEncoderLayer(1, hidden_dim1, hidden_dim2, num_heads)
        )
        for _ in range(num_layers - 1):
            self.encoder_layers.append(
                TransformerEncoderLayer(hidden_dim2, hidden_dim2, hidden_dim2, num_heads)
            )

        self.decoder_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.decoder_layers.append(
                TransformerDecoderLayer(hidden_dim2, hidden_dim1, hidden_dim2)
            )
        self.decoder_layers.append(
            TransformerDecoderLayer(1, hidden_dim1, hidden_dim2)
        )

    def forward(self, x, mask=None):
        # x: (batch, feature_dim)
        x = x.unsqueeze(-1)
        for layer in self.encoder_layers:
            x = layer(x, mask=mask)
        encoded_output = x
        for layer in self.decoder_layers:
            x = layer(x)
        reconstructed_output = x.squeeze(-1)
        return encoded_output, reconstructed_output


class ResidualBlock(nn.Module):
    def __init__(self, h_dim, h_fn=None, keep_prob=1.0):
        super().__init__()
        self.fc1 = nn.Linear(h_dim, h_dim)
        self.bn1 = nn.LayerNorm(h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.bn2 = nn.LayerNorm(h_dim)
        self.activation = get_activation_fn(h_fn) if h_fn else None
        self.dropout = nn.Dropout(p=1 - keep_prob) if keep_prob < 1.0 else None

    def forward(self, x):
        residual = x
        out = self.bn1(self.fc1(x))
        if self.activation:
            out = self.activation(out)
        if self.dropout:
            out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        if self.activation:
            out = self.activation(out)
        if self.dropout:
            out = self.dropout(out)
        return out + residual


def create_fc_net(input_dim, num_layers, h_dim, h_fn, o_dim, o_fn, keep_prob=1.0, use_resnet=False):
    layers = []
    if num_layers == 1:
        layers.append(nn.Linear(input_dim, o_dim))
        # LayerNorm(1) collapses scalar outputs to a learnable constant, which
        # breaks 1-D heads (e.g., binary classification probability logits/probs).
        if o_dim > 1:
            layers.append(nn.LayerNorm(o_dim))
        if o_fn:
            layers.append(get_activation_fn(o_fn))
        return nn.Sequential(*layers)

    layers.append(nn.Linear(input_dim, h_dim))
    layers.append(nn.LayerNorm(h_dim))
    if h_fn:
        layers.append(get_activation_fn(h_fn))
    if keep_prob < 1.0:
        layers.append(nn.Dropout(p=1 - keep_prob))

    for _ in range(1, num_layers - 1):
        if use_resnet:
            layers.append(ResidualBlock(h_dim, h_fn, keep_prob=keep_prob))
        else:
            layers.append(nn.Linear(h_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            if h_fn:
                layers.append(get_activation_fn(h_fn))
            if keep_prob < 1.0:
                layers.append(nn.Dropout(p=1 - keep_prob))

    layers.append(nn.Linear(h_dim, o_dim))
    # Skip output LayerNorm for scalar outputs for the same reason as above.
    if o_dim > 1:
        layers.append(nn.LayerNorm(o_dim))
    if o_fn:
        layers.append(get_activation_fn(o_fn))
    return nn.Sequential(*layers)


def create_outcome_specific_net(input_dim, num_layers, hidden_dim, activation_fn, output_dim, output_activation=None, keep_prob=1.0, use_resnet=False):
    return create_fc_net(
        input_dim=input_dim,
        num_layers=num_layers,
        h_dim=hidden_dim,
        h_fn=activation_fn,
        o_dim=output_dim,
        o_fn=output_activation,
        keep_prob=keep_prob,
        use_resnet=use_resnet,
    )


class ModelDeepHit_Multitask(nn.Module):
    def __init__(self, input_dims, network_settings, outcome_configs, autoencoder=None, writer=None):
        super().__init__()
        self.input_dims = input_dims
        self.network_settings = network_settings
        self.writer = writer

        self.x_dim = input_dims["x_dim"]
        self.num_Event = input_dims["num_Event"]
        self.num_Category = input_dims["num_Category"]

        self.h_dim_shared = network_settings["h_dim_shared"]
        self.h_dim_CS = network_settings["h_dim_CS"]
        self.num_layers_shared = network_settings["num_layers_shared"]
        self.num_layers_CS = network_settings["num_layers_CS"]
        self.active_fn = network_settings["active_fn"]
        self.keep_prob = network_settings["keep_prob"]
        self.ae_out_dim = network_settings.get("ae_out_dim", 16)

        self.autoencoder = autoencoder or DTA_AE(
            hidden_dim1=network_settings.get("ae_hidden_dim1", network_settings.get("ae_hidden_dim", 64)),
            hidden_dim2=network_settings.get("ae_hidden_dim2", self.ae_out_dim),
            num_heads=network_settings.get("ae_num_heads", 4),
            num_layers=network_settings.get("ae_num_layers", 1),
        )
        self.linear_layer = nn.Linear(self.ae_out_dim, 1)

        self.shared_net = create_fc_net(
            self.x_dim,
            self.num_layers_shared,
            self.h_dim_shared,
            self.active_fn,
            self.h_dim_shared,
            self.active_fn,
            keep_prob=self.keep_prob,
            use_resnet=True,
        )

        self.cs_nets = nn.ModuleList(
            [
                create_fc_net(
                    self.h_dim_shared + self.x_dim,
                    self.num_layers_CS,
                    self.h_dim_CS,
                    self.active_fn,
                    self.h_dim_CS,
                    self.active_fn,
                    keep_prob=self.keep_prob,
                    use_resnet=True,
                )
                for _ in range(self.num_Event)
            ]
        )

        self.outcome_pred_nets = nn.ModuleList(
            [
                create_outcome_specific_net(
                    input_dim=self.h_dim_shared + self.x_dim,
                    num_layers=self.num_layers_CS,
                    hidden_dim=self.h_dim_CS,
                    activation_fn=self.active_fn,
                    output_dim=config["output_dim"],
                    output_activation=config["output_activation"],
                    keep_prob=self.keep_prob,
                    use_resnet=True,
                )
                for config in outcome_configs
            ]
        )

        self.output_layer = nn.Linear(self.num_Event * self.h_dim_CS, self.num_Event * self.num_Category)
        self.softmax = nn.Softmax(dim=-1)
        # Kendall et al. uncertainty weighting (learned log variances) for:
        # [surv_ll, surv_rank, surv_cal] + outcome tasks
        self.loss_log_vars = nn.Parameter(torch.zeros(3 + len(outcome_configs), dtype=torch.float32))

    def forward(self, x, mask):
        ae_out, _ = self.autoencoder(x, mask=mask)
        ae_out = self.linear_layer(ae_out).squeeze(-1)
        self.ae_out = ae_out

        shared_out = self.shared_net(ae_out)
        h = torch.cat([ae_out, shared_out], dim=1)

        out_list = [cs_net(h) for cs_net in self.cs_nets]
        out = torch.cat(out_list, dim=1)
        out = F.dropout(out, p=1 - self.keep_prob, training=self.training)

        out = self.output_layer(out)                              # [N, E*T]
        out = torch.softmax(out, dim=1)                           # joint normalization over event*time
        out = out.view(-1, self.num_Event, self.num_Category)
        self.out = out

        outcome_preds = [net(h) for net in self.outcome_pred_nets]
        return self.out, outcome_preds, self.ae_out

    @staticmethod
    def _split_batch(batch):
        if len(batch) != 9:
            raise ValueError(f"Unexpected batch length: {len(batch)}")
        return batch

    def ae_loss(self):
        return torch.mean(torch.abs(self.ae_out))

    def get_regularization_loss(self):
        reg = torch.tensor(0.0, device=self.out.device)
        for p in self.parameters():
            reg = reg + torch.norm(p, p=2) ** 2
        return reg

    def loss_log_likelihood(self, k, fc_mask1):
        I_1 = torch.sign(k)
        tmp1 = torch.sum(torch.sum(fc_mask1 * self.out, dim=2), dim=1, keepdim=True)
        tmp1 = I_1 * log(tmp1)
        tmp2 = torch.sum(torch.sum(fc_mask1 * self.out, dim=2), dim=1, keepdim=True)
        tmp2 = (1.0 - I_1) * log(tmp2)
        return -torch.mean(tmp1 + tmp2)

    def loss_ranking(self, k, t, fc_mask2):
        sigma1 = 0.1
        eta = []
        for e in range(self.num_Event):
            one_vector = torch.ones_like(t)
            I_2 = (k == (e + 1)).float()
            I_2 = torch.diag(I_2.squeeze())
            tmp_e = self.out[:, e, :]

            R = torch.matmul(tmp_e, fc_mask2.T)
            diag_R = torch.diagonal(R).reshape(-1, 1)
            R = (diag_R - R).T

            T = F.relu(torch.sign(torch.matmul(one_vector, t.T) - torch.matmul(t, one_vector.T)))
            T = torch.matmul(I_2, T)

            tmp_eta = torch.mean(T * torch.exp(-R / sigma1), dim=1, keepdim=True)
            eta.append(tmp_eta)

        eta = torch.stack(eta, dim=1)
        eta = torch.mean(eta.reshape(-1, self.num_Event), dim=1, keepdim=True)
        return torch.sum(eta)

    def loss_calibration(self, k, fc_mask2):
        eta = []
        for e in range(self.num_Event):
            I_2 = (k == (e + 1)).float()
            tmp_e = self.out[:, e, :]
            r = torch.sum(tmp_e * fc_mask2, dim=1)
            tmp_eta = torch.mean((r - I_2) ** 2, dim=0, keepdim=True)
            eta.append(tmp_eta)
        eta = torch.stack(eta, dim=1)
        eta = torch.mean(eta.reshape(-1, self.num_Event), dim=1, keepdim=True)
        return torch.sum(eta)

    def mse_longitudinal(self, outcome_true, outcome_pred, missing_mask_fp, basis):
        outcome_true_time = outcome_true[:, :, 1]
        outcome_true_values = outcome_true[:, :, 0]
        outcome_true_values = torch.where(torch.isnan(outcome_true_values), torch.zeros_like(outcome_true_values), outcome_true_values)
        outcome_true_time = torch.where(torch.isnan(outcome_true_time), torch.zeros_like(outcome_true_time), outcome_true_time)

        outcome_pred = outcome_pred.unsqueeze(1)
        time_aware_weight = 0.5 ** outcome_true_time

        curve_prediction = torch.matmul(outcome_pred, basis).squeeze(1)
        loss_elementwise = time_aware_weight * (outcome_true_values - curve_prediction) ** 2
        denom = torch.clamp(missing_mask_fp.sum(), min=1.0)
        return (loss_elementwise * missing_mask_fp).sum() / denom

    def compute_loss(
        self,
        k,
        t,
        fc_mask1,
        fc_mask2,
        alpha,
        beta,
        gamma,
        delta,
        eta,
        fi,
        outcomes_true,
        outcome_preds,
        outcome_configs,
        missing_mask,
        basis,
    ):
        loss1 = self.loss_log_likelihood(k, fc_mask1)
        loss2 = self.loss_ranking(k, t, fc_mask2)
        loss3 = self.loss_calibration(k, fc_mask2)
        survival_loss = alpha * loss1 + beta * loss2 + gamma * loss3

        outcome_losses = []
        for i, config in enumerate(outcome_configs):
            task_type = config["task_type"]
            outcome_idx = int(config.get("outcome_idx", i))
            if task_type == "regression":
                observed = missing_mask[:, outcome_idx, 0, 0] > 0
                if torch.any(observed):
                    true_values = outcomes_true[:, outcome_idx, 0, 0][observed]
                    pred_values = outcome_preds[i].squeeze()[observed]
                    loss = torch.mean((true_values - pred_values) ** 2)
                else:
                    loss = torch.tensor(0.0, device=outcome_preds[i].device, dtype=outcome_preds[i].dtype)
            elif task_type == "binary_classification":
                observed = missing_mask[:, outcome_idx, 0, 0] > 0
                if torch.any(observed):
                    true_values = outcomes_true[:, outcome_idx, 0, 0][observed]
                    pred_values = outcome_preds[i].squeeze()[observed]
                    loss = F.binary_cross_entropy(pred_values, true_values)
                else:
                    loss = torch.tensor(0.0, device=outcome_preds[i].device, dtype=outcome_preds[i].dtype)
            elif task_type == "multiclass_classification":
                observed = missing_mask[:, outcome_idx, 0, 0] > 0
                if torch.any(observed):
                    true_values = outcomes_true[:, outcome_idx, 0, 0][observed]
                    pred_values = outcome_preds[i][observed]
                    loss = F.cross_entropy(pred_values, true_values.long())
                else:
                    loss = torch.tensor(0.0, device=outcome_preds[i].device, dtype=outcome_preds[i].dtype)
            elif task_type == "longitudinal_regression":
                loss = self.mse_longitudinal(
                    outcomes_true[:, outcome_idx, :, :],
                    outcome_preds[i],
                    missing_mask[:, outcome_idx, :, 0],
                    basis[:, outcome_idx, :, :],
                )
            else:
                raise ValueError(f"Unsupported task type: {task_type}")
            outcome_losses.append(loss)

        multitask_loss = torch.sum(torch.stack([w * l for w, l in zip(delta, outcome_losses)]))
        regularization_loss = self.get_regularization_loss()
        sparse_autoencoder_loss = self.ae_loss()
        return survival_loss + multitask_loss + eta * regularization_loss + fi * sparse_autoencoder_loss

    def adapt_shared_on_val(self, val_loader, outcome_configs, lr=1e-4, steps=50, device="cuda"):
        """Test-time training on validation data: update shared representation + longitudinal heads."""
        device = torch.device(device)
        # Avoid deepcopy(self): it can fail on non-leaf tensors.
        # Rebuild the module and load weights instead.
        model_ft = self.__class__(
            input_dims=self.input_dims,
            network_settings=self.network_settings,
            outcome_configs=outcome_configs,
            autoencoder=copy.deepcopy(self.autoencoder),
            writer=None,
        ).to(device)
        model_ft.load_state_dict(self.state_dict())
        model_ft.train()

        for p in model_ft.cs_nets.parameters():
            p.requires_grad = False
        for p in model_ft.output_layer.parameters():
            p.requires_grad = False

        for p in model_ft.shared_net.parameters():
            p.requires_grad = True
        for p in model_ft.linear_layer.parameters():
            p.requires_grad = True
        for p in model_ft.autoencoder.parameters():
            p.requires_grad = True

        for i, cfg in enumerate(outcome_configs):
            for p in model_ft.outcome_pred_nets[i].parameters():
                p.requires_grad = cfg["task_type"] in {"longitudinal_regression", "binary_classification"}

        trainable_params = [p for p in model_ft.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            return model_ft

        optimizer = torch.optim.Adam(trainable_params, lr=lr)

        for _ in range(steps):
            for batch in val_loader:
                batch = [b.to(device) for b in batch]
                (
                    x_mb,
                    _,
                    _,
                    _,
                    _,
                    outcomes,
                    missing_mask,
                    missing_mask_fp,
                    basis,
                ) = self._split_batch(batch)
                x_mb = torch.where(torch.isnan(x_mb), torch.zeros_like(x_mb), x_mb)

                optimizer.zero_grad()
                _, outcome_preds, _ = model_ft.forward(x_mb, missing_mask)

                loss = torch.tensor(0.0, device=device)
                for i, cfg in enumerate(outcome_configs):
                    if cfg["task_type"] == "longitudinal_regression":
                        outcome_idx = int(cfg.get("outcome_idx", i))
                        loss = loss + model_ft.mse_longitudinal(
                            outcomes[:, outcome_idx, :, :],
                            outcome_preds[i],
                            missing_mask_fp[:, outcome_idx, :, 0],
                            basis[:, outcome_idx, :, :],
                        )
                    elif cfg["task_type"] == "binary_classification":
                        outcome_idx = int(cfg.get("outcome_idx", i))
                        observed = missing_mask_fp[:, outcome_idx, 0, 0] > 0
                        if torch.any(observed):
                            probs = outcome_preds[i].squeeze()[observed]
                            y_true = outcomes[:, outcome_idx, 0, 0][observed]
                            loss = loss + F.binary_cross_entropy(probs, y_true)

                if loss.requires_grad:
                    loss.backward()
                    optimizer.step()

        return model_ft

    def _run_eval(self, loader, outcome_configs, device, use_ttt=True, ttt_lr=1e-4, ttt_steps=50):
        model_eval = self
        if use_ttt:
            model_eval = self.adapt_shared_on_val(
                val_loader=loader,
                outcome_configs=outcome_configs,
                lr=ttt_lr,
                steps=ttt_steps,
                device=device,
            )
        model_eval.eval()
        with torch.no_grad():
            batch = [b.to(device) for b in loader.dataset[:]]
            (
                x_val,
                k_val,
                t_val,
                _,
                _,
                outcomes_val,
                missing_mask_val,
                missing_mask_fp_val,
                basis_val,
            ) = self._split_batch(batch)

            cause_out_val, outcome_preds_val, _ = model_eval.forward(x_val, missing_mask_val)
            c_index_1 = overall_cause_specific_c_index(cause_out_val, k_val.flatten(), t_val.flatten(), num_causes_idx=0)
            _, ibs_1 = cause_specific_intergrated_brier_score(cause_out_val, t_val.flatten(), k_val.flatten(), num_causes_idx=0)
            if int(self.num_Event) >= 2:
                c_index_2 = overall_cause_specific_c_index(cause_out_val, k_val.flatten(), t_val.flatten(), num_causes_idx=1)
                _, ibs_2 = cause_specific_intergrated_brier_score(cause_out_val, t_val.flatten(), k_val.flatten(), num_causes_idx=1)
                surv_metrics = [c_index_1, c_index_2, ibs_1, ibs_2]
            else:
                surv_metrics = [c_index_1, ibs_1]

            outcomes_val = torch.where(torch.isnan(outcomes_val), torch.zeros_like(outcomes_val), outcomes_val)
            multitask_metrics = []
            for i, config in enumerate(outcome_configs):
                outcome_idx = int(config.get("outcome_idx", i))
                if config["task_type"] == "longitudinal_regression":
                    metric = model_eval.mse_longitudinal(
                        outcomes_val[:, outcome_idx, :, :],
                        outcome_preds_val[i],
                        missing_mask_fp_val[:, outcome_idx, :, 0],
                        basis_val[:, outcome_idx, :, :],
                    ).item()
                elif config["task_type"] == "binary_classification":
                    observed = missing_mask_fp_val[:, outcome_idx, 0, 0] > 0
                    if torch.any(observed):
                        probs = outcome_preds_val[i].squeeze()[observed]
                        y_true = outcomes_val[:, outcome_idx, 0, 0][observed]
                        y_pred = (probs >= 0.5).float()
                        metric = (y_pred == y_true).float().mean().item()
                    else:
                        metric = 0.0
                else:
                    observed = missing_mask_fp_val[:, outcome_idx, 0, 0] > 0
                    if torch.any(observed):
                        metric = torch.mean(
                            (
                                outcomes_val[:, outcome_idx, 0, 0][observed]
                                - outcome_preds_val[i].squeeze()[observed]
                            ) ** 2
                        ).item()
                    else:
                        metric = 0.0
                multitask_metrics.append(metric)
            return surv_metrics, multitask_metrics

    def pretrain_autoencoder(
        self,
        train_loader,
        epochs=50,
        lr=1e-3,
        sparse_weight=1e-4,
        patience=10,
        device="cuda",
        save_path=None,
    ):
        """Pretrain DTA_AE with masked reconstruction + sparse penalty."""
        device = torch.device(device)
        self.to(device)
        self.train()

        if save_path is None:
            save_path = DEFAULT_AE_SAVE_PATH
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        best_loss = float("inf")
        patience_counter = 0

        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=lr)

        for epoch in range(epochs):
            self.train()
            total_loss = 0.0

            for batch in train_loader:
                x_mb = batch[0].to(device)
                missing_mask = batch[6].to(device)  # baseline feature missing mask
                x_mb = torch.where(torch.isnan(x_mb), torch.zeros_like(x_mb), x_mb)

                optimizer.zero_grad()
                encoded, reconstructed = self.autoencoder(x_mb, mask=missing_mask)
                reconstruction_loss = torch.sum((x_mb - reconstructed) ** 2 * missing_mask) / torch.clamp(
                    torch.sum(missing_mask), min=1.0
                )
                sparse_loss = torch.mean(torch.abs(encoded))
                loss = reconstruction_loss + sparse_weight * sparse_loss
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())

            avg_loss = total_loss / max(len(train_loader), 1)
            print(f"[AE pretrain] Epoch [{epoch + 1}/{epochs}] Loss: {avg_loss:.6f}")

            if avg_loss + 1e-8 < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                torch.save(self.autoencoder.state_dict(), save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[AE pretrain] Early stopping at epoch {epoch + 1}.")
                    if os.path.exists(save_path):
                        self.autoencoder.load_state_dict(torch.load(save_path, map_location=device))
                    break

    def train_model(self, train_loader, val_loader, optimizer, alpha, beta, gamma, delta, eta, fi,
                    outcome_configs, epochs=300, patience=20, min_delta=1e-4, weights_on_metric=None,
                    device="cuda", use_ttt=True, ttt_lr=1e-4, ttt_steps=50,
                    pretrain_ae=False, ae_pretrain_epochs=50, ae_pretrain_lr=1e-3,
                    ae_pretrain_sparse_weight=1e-4, ae_pretrain_patience=10,
                    use_torchjd=True, adaptive_loss_weight=False, adaptive_ema=0.9,
                    adaptive_eps=1e-6, adaptive_min=0.2, adaptive_max=5.0,
                    uncertainty_weighting=False,
                    ae_pretrain_save_path=None,
                    best_model_save_path=None):
        device = torch.device(device)
        self.to(device)
        if adaptive_loss_weight and uncertainty_weighting:
            raise ValueError("Set only one of adaptive_loss_weight or uncertainty_weighting to True.")
        if uncertainty_weighting and use_torchjd:
            print("[Info] uncertainty_weighting=True: force use_torchjd=False for stable scalar objective.")
            use_torchjd = False
        if use_torchjd and (torchjd_mtl_backward is None or TorchJDUPGrad is None):
            raise ImportError("torchjd is not available. Install with `pip install torchjd`.")
        jd_aggregator = TorchJDUPGrad() if use_torchjd else None

        if pretrain_ae:
            self.pretrain_autoencoder(
                train_loader=train_loader,
                epochs=ae_pretrain_epochs,
                lr=ae_pretrain_lr,
                sparse_weight=ae_pretrain_sparse_weight,
                patience=ae_pretrain_patience,
                device=device,
                save_path=ae_pretrain_save_path,
            )

        self.best_weighted_metric = float("-inf")
        self.best_survival_metric = [0.0] * (4 if int(self.num_Event) >= 2 else 2)
        self.best_multitask_metrics = [0.0] * len(outcome_configs)
        patience_counter = 0
        self.val_history = []
        task_weight_names = ["surv_ll", "surv_rank", "surv_cal"] + [
            cfg.get("name", f"task_{i}") for i, cfg in enumerate(outcome_configs)
        ]
        base_task_weights = torch.tensor(
            [alpha, beta, gamma] + list(delta), dtype=torch.float32, device=device
        )
        adaptive_ema_losses = None
        latest_adaptive_weights = None

        best_model_path = best_model_save_path or DEFAULT_BEST_MODEL_PATH
        os.makedirs(os.path.dirname(best_model_path) or DEFAULT_SAVE_DIR, exist_ok=True)

        for epoch in range(epochs):
            self.train()
            total_loss = 0.0

            for batch in train_loader:
                batch = [b.to(device) for b in batch]
                (
                    x_mb,
                    k_mb,
                    t_mb,
                    m1_mb,
                    m2_mb,
                    outcomes_true,
                    missing_mask,
                    missing_mask_fp,
                    basis_mb,
                ) = self._split_batch(batch)
                x_mb = torch.where(torch.isnan(x_mb), torch.zeros_like(x_mb), x_mb)

                optimizer.zero_grad()
                cause_out, outcome_preds, shared_features = self.forward(x_mb, missing_mask)

                loss1 = self.loss_log_likelihood(k_mb, m1_mb)
                loss2 = self.loss_ranking(k_mb, t_mb, m2_mb)
                loss3 = self.loss_calibration(k_mb, m2_mb)

                outcome_losses = []
                for i, config in enumerate(outcome_configs):
                    task_type = config["task_type"]
                    outcome_idx = int(config.get("outcome_idx", i))
                    if task_type == "regression":
                        observed = missing_mask_fp[:, outcome_idx, 0, 0] > 0
                        if torch.any(observed):
                            true_values = outcomes_true[:, outcome_idx, 0, 0][observed]
                            pred_values = outcome_preds[i].squeeze()[observed]
                            task_loss = torch.mean((true_values - pred_values) ** 2)
                        else:
                            task_loss = torch.tensor(0.0, device=outcome_preds[i].device, dtype=outcome_preds[i].dtype)
                    elif task_type == "binary_classification":
                        observed = missing_mask_fp[:, outcome_idx, 0, 0] > 0
                        if torch.any(observed):
                            probs = outcome_preds[i].squeeze()[observed]
                            y_true = outcomes_true[:, outcome_idx, 0, 0][observed]
                            task_loss = F.binary_cross_entropy(probs, y_true)
                        else:
                            task_loss = torch.tensor(0.0, device=outcome_preds[i].device, dtype=outcome_preds[i].dtype)
                    elif task_type == "multiclass_classification":
                        observed = missing_mask_fp[:, outcome_idx, 0, 0] > 0
                        if torch.any(observed):
                            true_values = outcomes_true[:, outcome_idx, 0, 0][observed]
                            pred_values = outcome_preds[i][observed]
                            task_loss = F.cross_entropy(pred_values, true_values.long())
                        else:
                            task_loss = torch.tensor(0.0, device=outcome_preds[i].device, dtype=outcome_preds[i].dtype)
                    elif task_type == "longitudinal_regression":
                        task_loss = self.mse_longitudinal(
                            outcomes_true[:, outcome_idx, :, :],
                            outcome_preds[i],
                            missing_mask_fp[:, outcome_idx, :, 0],
                            basis_mb[:, outcome_idx, :, :],
                        )
                    else:
                        raise ValueError(f"Unsupported task type: {task_type}")
                    outcome_losses.append(task_loss)

                task_losses_raw = [loss1, loss2, loss3] + outcome_losses
                if adaptive_loss_weight:
                    current_losses = torch.stack([l.detach().float() for l in task_losses_raw])
                    if adaptive_ema_losses is None:
                        adaptive_ema_losses = current_losses
                    else:
                        adaptive_ema_losses = adaptive_ema * adaptive_ema_losses + (1.0 - adaptive_ema) * current_losses
                    inv = 1.0 / (adaptive_ema_losses + adaptive_eps)
                    multipliers = inv / torch.mean(inv)
                    multipliers = torch.clamp(multipliers, min=adaptive_min, max=adaptive_max)
                    effective_task_weights = base_task_weights * multipliers
                    latest_adaptive_weights = effective_task_weights.detach().cpu().numpy()
                else:
                    effective_task_weights = base_task_weights

                surv_w = effective_task_weights[:3]
                outcome_w = effective_task_weights[3:]
                weighted_outcome_losses = [w * l for w, l in zip(outcome_w, outcome_losses)]
                if uncertainty_weighting:
                    if self.loss_log_vars.shape[0] != len(task_losses_raw):
                        raise ValueError(
                            f"loss_log_vars size {self.loss_log_vars.shape[0]} != task count {len(task_losses_raw)}"
                        )
                    task_stack = torch.stack(task_losses_raw)
                    base_scaled = base_task_weights * task_stack
                    precision = torch.exp(-self.loss_log_vars)
                    uw_terms = precision * base_scaled + self.loss_log_vars
                    uncertainty_weighted_loss = torch.sum(uw_terms)

                if use_torchjd:
                    reg_loss = eta * self.get_regularization_loss()
                    sparse_loss = fi * self.ae_loss()
                    objective_losses = (
                        [surv_w[0] * loss1, surv_w[1] * loss2, surv_w[2] * loss3]
                        + weighted_outcome_losses
                        + [reg_loss, sparse_loss]
                    )
                    shared_prefixes = ["autoencoder.", "linear_layer.", "shared_net."]
                    shared_params = [
                        p for name, p in self.named_parameters()
                        if p.requires_grad and any(name.startswith(prefix) for prefix in shared_prefixes)
                    ]
                    task_params = [
                        p for name, p in self.named_parameters()
                        if p.requires_grad and not any(name.startswith(prefix) for prefix in shared_prefixes)
                    ]
                    tasks_params = [task_params for _ in objective_losses]
                    _torchjd_mtl_backward_compat(
                        objective_losses,
                        shared_features,
                        jd_aggregator,
                        tasks_params,
                        shared_params,
                    )
                    optimizer.step()
                    total_loss += float(sum([obj.detach().item() for obj in objective_losses]))
                else:
                    reg_loss = eta * self.get_regularization_loss()
                    sparse_loss = fi * self.ae_loss()
                    if uncertainty_weighting:
                        loss = uncertainty_weighted_loss + reg_loss + sparse_loss
                    else:
                        survival_loss = surv_w[0] * loss1 + surv_w[1] * loss2 + surv_w[2] * loss3
                        if len(weighted_outcome_losses) > 0:
                            multitask_loss = torch.sum(torch.stack(weighted_outcome_losses))
                        else:
                            multitask_loss = torch.tensor(0.0, device=device)
                        loss = survival_loss + multitask_loss + reg_loss + sparse_loss
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss.item())

            avg_loss = total_loss / max(len(train_loader), 1)
            print(f"Epoch [{epoch + 1}/{epochs}] Loss: {avg_loss:.4f}")
            if adaptive_loss_weight and latest_adaptive_weights is not None:
                msg = ", ".join(
                    [f"{n}={w:.3f}" for n, w in zip(task_weight_names, latest_adaptive_weights.tolist())]
                )
                print(f"[AdaptiveWeight] {msg}")
            if uncertainty_weighting:
                with torch.no_grad():
                    precision = torch.exp(-self.loss_log_vars.detach())
                    eff = (base_task_weights * precision).detach().cpu().numpy().tolist()
                msg = ", ".join([f"{n}={w:.3f}" for n, w in zip(task_weight_names, eff)])
                print(f"[UncertaintyWeight] {msg}")

            if (epoch + 1) % 5 == 0:
                surv_metrics, multitask_metrics = self._run_eval(
                    val_loader,
                    outcome_configs,
                    device,
                    use_ttt=use_ttt,
                    ttt_lr=ttt_lr,
                    ttt_steps=ttt_steps,
                )
                task_names = [cfg.get("name", f"task_{i}") for i, cfg in enumerate(outcome_configs)]

                if weights_on_metric:
                    n_surv_metrics = len(surv_metrics)
                    expected_weights = n_surv_metrics + len(multitask_metrics)
                    if len(weights_on_metric) < expected_weights:
                        raise ValueError(
                            f"weights_on_metric length {len(weights_on_metric)} is smaller than "
                            f"required {expected_weights} (={n_surv_metrics} survival + {len(multitask_metrics)} multitask)"
                        )
                    weighted_metric = 0.0
                    weighted_detail = []
                    for task_metrics, weight in zip(surv_metrics, weights_on_metric[:n_surv_metrics]):
                        weighted_metric += weight * task_metrics
                        weighted_detail.append(weight * task_metrics)
                    for task_metrics, weight in zip(multitask_metrics, weights_on_metric[n_surv_metrics:]):
                        weighted_metric += weight * task_metrics
                        weighted_detail.append(weight * task_metrics)
                else:
                    weighted_metric = surv_metrics[0]
                    weighted_detail = []

                multitask_msg = ", ".join(
                    [f"{name}={metric:.4f}" for name, metric in zip(task_names, multitask_metrics)]
                )

                if int(self.num_Event) >= 2 and len(surv_metrics) >= 4:
                    surv_msg = (
                        f"Validation Event1 C={surv_metrics[0]:.4f}, IBS={surv_metrics[2]:.4f}; "
                        f"Event2 C={surv_metrics[1]:.4f}, IBS={surv_metrics[3]:.4f}; "
                    )
                else:
                    surv_msg = f"Validation Event1 C={surv_metrics[0]:.4f}, IBS={surv_metrics[1]:.4f}; "
                print(f"{surv_msg}Multitask[{multitask_msg}]; Weighted={weighted_metric:.4f}")
                if weighted_detail:
                    print(f"Weighted components: {[round(v, 6) for v in weighted_detail]}")

                self.val_history.append(
                    {
                        "epoch": epoch + 1,
                        "survival_metrics": surv_metrics,
                        "multitask_metrics": {n: m for n, m in zip(task_names, multitask_metrics)},
                        "weighted_metric": float(weighted_metric),
                    }
                )
                self.latest_val_metrics = self.val_history[-1]

                if weighted_metric > self.best_weighted_metric + min_delta:
                    self.best_weighted_metric = weighted_metric
                    self.best_survival_metric = surv_metrics
                    self.best_multitask_metrics = multitask_metrics
                    torch.save(self.state_dict(), best_model_path)
                    patience_counter = 0
                    print(f"Best model updated at epoch {epoch + 1}")
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    self.load_state_dict(torch.load(best_model_path, map_location=device))
                    break

    def predict(self, x, missing_mask, device="cuda"):
        device = torch.device(device)
        self.eval()
        with torch.no_grad():
            x = x.to(device)
            missing_mask = missing_mask.to(device)
            cause_out, outcome_preds, _ = self.forward(x, missing_mask)
        return cause_out, outcome_preds


def build_tensor_dataset_and_loader(
    X,
    E,
    T,
    mask1,
    mask2,
    outcome_matrix,
    mask_miss,
    mask_miss_pred,
    batch_basis,
    batch_size=64,
    shuffle=True,
):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(E, dtype=torch.float32),
        torch.tensor(T, dtype=torch.float32),
        torch.tensor(mask1, dtype=torch.float32),
        torch.tensor(mask2, dtype=torch.float32),
        torch.tensor(outcome_matrix, dtype=torch.float32),
        torch.tensor(mask_miss, dtype=torch.float32),
        torch.tensor(mask_miss_pred, dtype=torch.float32),
        torch.tensor(batch_basis, dtype=torch.float32),
    )
    return ds, DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
