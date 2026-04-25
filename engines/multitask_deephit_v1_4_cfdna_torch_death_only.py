import json
import os
import random
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
LEGACY_PROJECT_DIR = PROJECT_DIR
DATA_DIR = os.path.join(PROJECT_DIR, "data")
if SCRIPT_DIR not in sys.path:
	sys.path.append(SCRIPT_DIR)
if PROJECT_DIR not in sys.path:
	sys.path.append(PROJECT_DIR)

import multitask_deephit_v1_3_cfdna_torch_death_only as legacy
from checkpoint_paths import (
	DEFAULT_AE_CHECKPOINT_PATH,
	DEFAULT_BEST_MODEL_CHECKPOINT_PATH,
	ensure_checkpoint_dirs,
)

legacy.DEFAULT_AE_SAVE_PATH = DEFAULT_AE_CHECKPOINT_PATH
legacy.DEFAULT_BEST_MODEL_PATH = DEFAULT_BEST_MODEL_CHECKPOINT_PATH

TIME_BIN_EDGES_YEARS = legacy.TIME_BIN_EDGES_YEARS
TIME_BIN_REPRESENTATIVES_YEARS = legacy.TIME_BIN_REPRESENTATIVES_YEARS
EVENT_IBS_HORIZON_YEARS = {0: 2.0, 1: 2.0}

BASELINE_PKL_FILENAME = "demo_survival_supplementary_no30_pft_io_death_only_v3.pkl"
BASELINE_CSV_FILENAME = "demo_survival_supplementary_no30_death_only_v3.csv"
V2_COLUMN_RENAMES = {
	"log2_rdcfDNA_(Copies/mL)": "log2_rdcfDNA_.Copies.mL.",
}
EMBEDDING_MIN_CLASSES = 3

CATEGORY_COLUMNS = [
	"final_TRANSPLANT_TYPE",
	"SITE",
	"SEX",
	"ETHNICITY",
	"NATIVE_LUNG_DISEASE_Coded",
	"RACE",
	"DONOR_CMV_SEROLOGY",
	"DONOR_GENDER",
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
	"SMOKING_HISTORY_RECIPIENT",
	"Race_mismatch",
	"Blood_type_mismatch",
]

BINARY_OUTCOME_COLUMNS = ["severe_ACR", "ever_clinical_AMR", "BLAD"]
LONGITUDINAL_OUTCOME_COLUMNS = ["FEV1_PCT_matrix"]

# FEV1_PCT_matrix stores visit time in months (e.g. 0.53, 1.0, 3.93, ..., 12.0).
BSPLINE_TIME_MIN_MONTHS = 0.0
BSPLINE_TIME_MAX_MONTHS = 12.0
BSPLINE_DEGREE_DEFAULT = 3
BSPLINE_INTERNAL_KNOTS_MONTHS: Tuple[float, ...] = (3.0, 9.0)
BSPLINE_NUM_INTERNAL_KNOTS_DEFAULT = len(BSPLINE_INTERNAL_KNOTS_MONTHS)

# Backward-compatible aliases for older imports.
BSPLINE_TIME_MIN_YEARS = BSPLINE_TIME_MIN_MONTHS
BSPLINE_TIME_MAX_YEARS = BSPLINE_TIME_MAX_MONTHS
BSPLINE_INTERNAL_KNOTS_YEARS = BSPLINE_INTERNAL_KNOTS_MONTHS


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
	if hasattr(torch.backends, "cudnn"):
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False


def get_time_bin_representatives_years(n_bins: int) -> np.ndarray:
	return legacy.get_time_bin_representatives_years(n_bins)


def _rmst_5y_from_cause_distribution(pred_event: np.ndarray, tau_years: float = 5.0) -> np.ndarray:
	edges = TIME_BIN_EDGES_YEARS.astype(np.float32)
	usable_bins = int(np.searchsorted(edges, tau_years, side="left") + 1)
	usable_bins = min(usable_bins, pred_event.shape[1], edges.shape[0])
	widths = np.diff(np.concatenate(([0.0], edges[:usable_bins]))).astype(np.float32)
	cumulative_event = np.cumsum(pred_event[:, :usable_bins], axis=1)
	survival_before_bin = np.concatenate(
		[np.ones((pred_event.shape[0], 1), dtype=np.float32), 1.0 - cumulative_event[:, :-1]],
		axis=1,
	)
	return np.sum(survival_before_bin * widths.reshape(1, -1), axis=1)


def overall_cause_specific_c_index(pred, event, time, num_causes_idx):
	out = pred[:, num_causes_idx, :].detach().cpu().numpy()
	rmst_5y = _rmst_5y_from_cause_distribution(out, tau_years=5.0)
	risk_score = -rmst_5y
	event_np = event.detach().cpu().numpy().squeeze()
	event_indicator_bool = event_np == (num_causes_idx + 1)
	try:
		c_index = legacy.concordance_index_censored(
			event_indicator_bool.squeeze(),
			time.detach().cpu().numpy().squeeze(),
			risk_score.squeeze(),
		)[0]
	except Exception:
		c_index = -1
	return c_index


legacy.overall_cause_specific_c_index = overall_cause_specific_c_index
legacy.EVENT_IBS_HORIZON_YEARS = EVENT_IBS_HORIZON_YEARS


DTA_AE = legacy.DTA_AE
build_tensor_dataset_and_loader = legacy.build_tensor_dataset_and_loader
cause_specific_intergrated_brier_score = legacy.cause_specific_intergrated_brier_score


def factorize_df_for_embeddings(df: pd.DataFrame) -> Tuple[np.ndarray, List[int]]:
	if df.shape[1] == 0:
		return np.zeros((len(df), 0), dtype=np.float32), []
	encoded_cols: List[np.ndarray] = []
	cardinalities: List[int] = []
	for col in df.columns:
		codes, uniques = pd.factorize(df[col], sort=True)
		codes = codes.astype(np.int64) + 1
		codes[codes < 0] = 0
		encoded_cols.append(codes)
		cardinalities.append(int(len(uniques) + 1))
	encoded = np.stack(encoded_cols, axis=1).astype(np.float32)
	return encoded, cardinalities


def split_categorical_feature_columns(
	df: pd.DataFrame,
	categorical_columns: Sequence[str],
	embedding_min_classes: int = EMBEDDING_MIN_CLASSES,
) -> Tuple[List[str], List[str]]:
	passthrough_columns: List[str] = []
	embedding_columns: List[str] = []
	for col in categorical_columns:
		n_classes = int(df[col].nunique(dropna=True))
		if n_classes >= embedding_min_classes:
			embedding_columns.append(col)
		else:
			passthrough_columns.append(col)
	return passthrough_columns, embedding_columns


def build_bspline_knots(
	degree: int = BSPLINE_DEGREE_DEFAULT,
	time_min: float = BSPLINE_TIME_MIN_MONTHS,
	time_max: float = BSPLINE_TIME_MAX_MONTHS,
	num_internal_knots: int = BSPLINE_NUM_INTERNAL_KNOTS_DEFAULT,
	internal_knots: Optional[Sequence[float]] = BSPLINE_INTERNAL_KNOTS_MONTHS,
) -> np.ndarray:
	if internal_knots is None:
		internal_knots_arr = np.linspace(time_min, time_max, int(num_internal_knots) + 2)[1:-1]
	else:
		internal_knots_arr = np.asarray(list(internal_knots), dtype=np.float64)
		if internal_knots_arr.ndim != 1:
			raise ValueError("internal_knots must be a 1D sequence.")
		if internal_knots_arr.size > 0:
			internal_knots_arr = np.sort(internal_knots_arr)
			if internal_knots_arr[0] <= time_min or internal_knots_arr[-1] >= time_max:
				raise ValueError("All internal_knots must lie strictly inside (time_min, time_max).")

	return np.concatenate(
		(
			np.repeat(time_min, degree + 1),
			internal_knots_arr,
			np.repeat(time_max, degree + 1),
		)
	)


def build_bspline_batch_basis(
	x_time: np.ndarray,
	bspline_degree: int = BSPLINE_DEGREE_DEFAULT,
	bspline_time_min: float = BSPLINE_TIME_MIN_MONTHS,
	bspline_time_max: float = BSPLINE_TIME_MAX_MONTHS,
	bspline_num_internal_knots: int = BSPLINE_NUM_INTERNAL_KNOTS_DEFAULT,
	bspline_internal_knots: Optional[Sequence[float]] = BSPLINE_INTERNAL_KNOTS_MONTHS,
) -> np.ndarray:
	batch_size, covariate_dim, _ = x_time.shape
	k = int(bspline_degree)
	t_knots = build_bspline_knots(
		degree=k,
		time_min=float(bspline_time_min),
		time_max=float(bspline_time_max),
		num_internal_knots=int(bspline_num_internal_knots),
		internal_knots=bspline_internal_knots,
	)

	num_basis = len(t_knots) - (k + 1)
	c = np.eye(num_basis)
	b_spline_basis = [legacy.BSpline(t_knots, c[i], k) for i in range(num_basis)]

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

	return np.stack(batch_basis, axis=0).astype(np.float32)


def import_dataset_cfdna_sim_outcome(
	norm_mode: str = "standard",
	bspline_degree: int = BSPLINE_DEGREE_DEFAULT,
	bspline_time_min: float = BSPLINE_TIME_MIN_MONTHS,
	bspline_time_max: float = BSPLINE_TIME_MAX_MONTHS,
	bspline_num_internal_knots: int = BSPLINE_NUM_INTERNAL_KNOTS_DEFAULT,
	bspline_internal_knots: Optional[Sequence[float]] = BSPLINE_INTERNAL_KNOTS_MONTHS,
):
	df_with_miss = pd.read_pickle(
		os.path.join(DATA_DIR, BASELINE_PKL_FILENAME)
	)
	df_with_miss = df_with_miss.rename(columns=V2_COLUMN_RENAMES)
	df_with_miss = df_with_miss[df_with_miss["event_time"] > 30].copy()
	df_baseline_with_miss = df_with_miss.drop(columns=LONGITUDINAL_OUTCOME_COLUMNS + BINARY_OUTCOME_COLUMNS).copy()
	df_long_for_pred = df_with_miss[LONGITUDINAL_OUTCOME_COLUMNS].copy()
	df_baseline_imputed = pd.read_csv(
		os.path.join(DATA_DIR, BASELINE_CSV_FILENAME)
	)
	df_baseline_imputed = df_baseline_imputed.rename(columns=V2_COLUMN_RENAMES)
	df_baseline_imputed = df_baseline_imputed[df_baseline_imputed["event_time"] > 30].copy()

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
	df_baseline_with_miss = df_with_miss.drop(columns=LONGITUDINAL_OUTCOME_COLUMNS + BINARY_OUTCOME_COLUMNS).copy()
	df_long_for_pred = df_with_miss[LONGITUDINAL_OUTCOME_COLUMNS].copy()

	subject_numbers = df_baseline_imputed["SUBJECT_NUMBER"].astype(str).to_numpy()
	available_category_columns = [
		col for col in CATEGORY_COLUMNS
		if col in df_baseline_imputed.columns and col in df_baseline_with_miss.columns
	]
	passthrough_category_columns, embedding_category_columns = split_categorical_feature_columns(
		df_baseline_imputed,
		available_category_columns,
	)

	numerical_columns = [
		col for col in df_baseline_imputed.columns
		if col not in {"SUBJECT_NUMBER", "event_time", "event", *available_category_columns}
		and col in df_baseline_with_miss.columns
	]
	df_numerical = df_baseline_imputed[numerical_columns].copy()
	df_numerical_scaled = legacy.f_get_Normalization(df_numerical.to_numpy(dtype=np.float32), norm_mode)
	df_numerical_mean = np.array(df_numerical.mean()).reshape(-1, 1)
	df_numerical_std = np.array(df_numerical.std()).reshape(-1, 1)
	df_numerical_mean_std = np.concatenate((df_numerical_mean, df_numerical_std), axis=1)

	data_parts = [df_numerical_scaled.astype(np.float32)]
	if passthrough_category_columns:
		data_parts.append(df_baseline_imputed[passthrough_category_columns].to_numpy(dtype=np.float32))
	X_cat, cat_cardinalities = factorize_df_for_embeddings(df_baseline_imputed[embedding_category_columns].copy())
	if X_cat.shape[1] > 0:
		data_parts.append(X_cat)
	data = np.concatenate(data_parts, axis=1).astype(np.float32)
	num_numerical = int(df_numerical_scaled.shape[1] + len(passthrough_category_columns))

	label = np.asarray(df_baseline_imputed[["event"]])
	time_days = np.asarray(df_baseline_imputed[["event_time"]], dtype=np.float32)
	time = legacy.discretize_time_to_custom_bins(time_days)

	num_category = int(TIME_BIN_REPRESENTATIVES_YEARS.shape[0])
	num_event = int(len(np.unique(label)) - 1)
	mask1 = legacy.f_get_fc_mask2(time, label, num_event, num_category)
	mask2 = legacy.f_get_fc_mask3(time, -1, num_category)

	df_miss_parts = [df_baseline_with_miss[numerical_columns].copy()]
	if passthrough_category_columns:
		df_miss_parts.append(df_baseline_with_miss[passthrough_category_columns].copy())
	if embedding_category_columns:
		df_miss_parts.append(df_baseline_with_miss[embedding_category_columns].copy())
	df_miss_all = pd.concat(df_miss_parts, axis=1)
	mask_miss = legacy.missing_mask_from_df(df_miss_all).astype(np.float32)

	outcome_matrix_pred_long = np.array(
		df_long_for_pred.drop(columns=["SUBJECT_NUMBER"], errors="ignore").to_numpy().tolist(),
		dtype=np.float64,
	)
	n_longitudinal = outcome_matrix_pred_long.shape[1]
	binary_raw = df_with_miss[BINARY_OUTCOME_COLUMNS].to_numpy(dtype=np.float64)
	n_samples, _, t_dim, _ = outcome_matrix_pred_long.shape
	binary_tensor = np.full((n_samples, len(BINARY_OUTCOME_COLUMNS), t_dim, 2), np.nan, dtype=np.float64)
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
	batch_basis = build_bspline_batch_basis(
		x_time=x_time,
		bspline_degree=bspline_degree,
		bspline_time_min=bspline_time_min,
		bspline_time_max=bspline_time_max,
		bspline_num_internal_knots=bspline_num_internal_knots,
		bspline_internal_knots=bspline_internal_knots,
	)

	return (
		data.astype("float32"),
		label.astype("float32"),
		time.astype("float32"),
		mask1.astype("float32"),
		mask2.astype("float32"),
		num_category,
		num_event,
		mask_miss.astype("float32"),
		outcome_matrix_scaled.astype("float32"),
		mask_miss_pred.astype("float32"),
		batch_basis.astype("float32"),
		df_numerical_mean_std.astype("float32"),
		long_mean_std.astype("float32"),
		subject_numbers,
		cat_cardinalities,
		num_numerical,
		list(available_category_columns),
	)


def prepare_input_dims(
	X: np.ndarray,
	num_event: int,
	num_category: int,
	num_numerical: int,
	cat_cardinalities: Sequence[int],
	cat_embedding_dim: int = 4,
) -> Dict:
	effective_x_dim = int(num_numerical + len(cat_cardinalities) * cat_embedding_dim)
	return {
		"x_dim": effective_x_dim,
		"raw_x_dim": int(X.shape[1]),
		"num_numerical": int(num_numerical),
		"cat_cardinalities": [int(v) for v in cat_cardinalities],
		"cat_embedding_dim": int(cat_embedding_dim),
		"num_Event": int(num_event),
		"num_Category": int(num_category),
	}


def outcome_metric_name(config: Dict) -> str:
	if config["task_type"] == "longitudinal_regression":
		return f"{config['name']}_MSE"
	if config["task_type"] == "binary_classification":
		return f"{config['name']}_ACC"
	return config["name"]


class ModelDeepHit_Multitask(legacy.ModelDeepHit_Multitask):
	def __init__(self, input_dims, network_settings, outcome_configs, autoencoder=None, writer=None):
		self.raw_x_dim = int(input_dims["raw_x_dim"])
		self.num_numerical = int(input_dims["num_numerical"])
		self.cat_cardinalities = [int(v) for v in input_dims.get("cat_cardinalities", [])]
		self.cat_embedding_dim = int(input_dims.get("cat_embedding_dim", 4))
		self.outcome_configs = list(outcome_configs)
		effective_input_dims = dict(input_dims)
		effective_input_dims["x_dim"] = int(
			self.num_numerical + len(self.cat_cardinalities) * self.cat_embedding_dim
		)
		super().__init__(effective_input_dims, network_settings, outcome_configs, autoencoder=autoencoder, writer=writer)
		self.cat_embeddings = nn.ModuleList(
			[nn.Embedding(cardinality, self.cat_embedding_dim) for cardinality in self.cat_cardinalities]
		)

	def _prepare_features(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		x_num = x[:, : self.num_numerical]
		mask_num = mask[:, : self.num_numerical]
		if not self.cat_cardinalities:
			return x_num, mask_num

		x_cat = x[:, self.num_numerical : self.num_numerical + len(self.cat_cardinalities)].long().clamp_min(0)
		mask_cat = mask[:, self.num_numerical : self.num_numerical + len(self.cat_cardinalities)]
		embedded_parts = []
		embedded_masks = []
		for idx, embedding in enumerate(self.cat_embeddings):
			emb = embedding(x_cat[:, idx])
			embedded_parts.append(emb)
			embedded_masks.append(mask_cat[:, idx : idx + 1].expand(-1, self.cat_embedding_dim))

		x_all = torch.cat([x_num] + embedded_parts, dim=1)
		mask_all = torch.cat([mask_num] + embedded_masks, dim=1)
		return x_all, mask_all

	def forward(self, x, mask):
		x_prepared, mask_prepared = self._prepare_features(x, mask)
		ae_out, _ = self.autoencoder(x_prepared, mask=mask_prepared)
		ae_out = self.linear_layer(ae_out).squeeze(-1)
		self.ae_out = ae_out

		shared_out = self.shared_net(ae_out)
		h = torch.cat([ae_out, shared_out], dim=1)

		out_list = [cs_net(h) for cs_net in self.cs_nets]
		out = torch.cat(out_list, dim=1)
		out = torch.nn.functional.dropout(out, p=1 - self.keep_prob, training=self.training)
		out = self.output_layer(out)
		out = torch.softmax(out, dim=1)
		out = out.view(-1, self.num_Event, self.num_Category)
		self.out = out
		outcome_preds = [net(h) for net in self.outcome_pred_nets]
		return self.out, outcome_preds, self.ae_out

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
		device = torch.device(device)
		self.to(device)
		self.train()
		ensure_checkpoint_dirs()
		if save_path is None:
			save_path = DEFAULT_AE_CHECKPOINT_PATH
		os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
		best_loss = float("inf")
		patience_counter = 0
		optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=lr)

		for epoch in range(epochs):
			self.train()
			total_loss = 0.0
			for batch in train_loader:
				x_mb = batch[0].to(device)
				missing_mask = batch[6].to(device)
				x_prepared, mask_prepared = self._prepare_features(x_mb, missing_mask)
				optimizer.zero_grad()
				encoded, reconstructed = self.autoencoder(x_prepared, mask=mask_prepared)
				reconstruction_loss = torch.sum((x_prepared - reconstructed) ** 2 * mask_prepared) / torch.clamp(
					torch.sum(mask_prepared), min=1.0
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
