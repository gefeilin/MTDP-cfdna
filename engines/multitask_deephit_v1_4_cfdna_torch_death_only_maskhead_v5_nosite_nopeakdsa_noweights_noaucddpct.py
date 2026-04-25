"""cfDNA v1.4/v5 model variant with missing-mask augmented heads.

This variant keeps the current v1.4 data pipeline, autoencoder, and training
API unchanged, but augments:

- survival head input
- longitudinal head input

with the prepared baseline missing-mask representation.

Binary intermediate heads keep the original input representation.

This v5 variant excludes:
- SITE
- Peak_DSA_strength
- final_WEIGHT_KG_TX
- DONOR_WT
- AUC_ddcfDNA_PCT_1_30
from predictors.
"""

from multitask_deephit_v1_4_cfdna_torch_death_only import *  # noqa: F401,F403
import multitask_deephit_v1_4_cfdna_torch_death_only as base_engine
from torch.utils.data import DataLoader, TensorDataset


BASELINE_PKL_FILENAME = "demo_survival_supplementary_no30_pft_io_death_only_v5.pkl"
BASELINE_CSV_FILENAME = "demo_survival_supplementary_no30_death_only_v5.csv"
CATEGORY_COLUMNS = [
    column for column in base_engine.CATEGORY_COLUMNS
    if column not in {"SITE", "Peak_DSA_strength"}
]
EXCLUDED_PREDICTOR_COLUMNS = {
    "SITE",
    "Peak_DSA_strength",
    "final_WEIGHT_KG_TX",
    "DONOR_WT",
    "AUC_ddcfDNA_PCT_1_30",
}
BSPLINE_TIME_MAX_MONTHS = 15.0
BSPLINE_TIME_MAX_YEARS = BSPLINE_TIME_MAX_MONTHS

# Keep the delegated base-engine data import path aligned with this variant.
base_engine.BASELINE_PKL_FILENAME = BASELINE_PKL_FILENAME
base_engine.BASELINE_CSV_FILENAME = BASELINE_CSV_FILENAME
base_engine.CATEGORY_COLUMNS = CATEGORY_COLUMNS


DAYS_PER_MONTH = 365.25 / 12.0
DEFAULT_LONGITUDINAL_PRETRAIN_CHECKPOINT_PATH = os.path.join(
    base_engine.SCRIPT_DIR, "saved_model", "longitudinal_pretrain_best_model.pt"
)


def _load_time_days_for_subject_numbers(subject_numbers):
    df = pd.read_csv(os.path.join(DATA_DIR, BASELINE_CSV_FILENAME)).copy()
    if V2_COLUMN_RENAMES:
        df = df.rename(columns={k: v for k, v in V2_COLUMN_RENAMES.items() if k in df.columns})
    df = df[df["event_time"] > 30].copy()
    df["SUBJECT_NUMBER"] = df["SUBJECT_NUMBER"].astype(str)
    time_series = df.drop_duplicates(subset=["SUBJECT_NUMBER"]).set_index("SUBJECT_NUMBER")["event_time"]
    ordered = time_series.reindex(pd.Index(np.asarray(subject_numbers).astype(str)))
    if ordered.isna().any():
        missing_subjects = ordered[ordered.isna()].index.tolist()
        raise ValueError(f"Missing event_time for subjects: {missing_subjects[:5]}")
    return ordered.to_numpy(dtype=np.float32).reshape(-1, 1)


def _coerce_longitudinal_cell(cell, target_rows):
    if cell is None:
        return np.full((target_rows, 2), np.nan, dtype=np.float64)

    arr = np.asarray(cell, dtype=np.float64)
    if arr.ndim == 0:
        scalar = float(arr)
        if np.isnan(scalar):
            return np.full((target_rows, 2), np.nan, dtype=np.float64)
        raise ValueError(f"Unexpected scalar longitudinal payload: {cell!r}")

    if arr.ndim == 1:
        if arr.size == 0:
            arr = np.empty((0, 2), dtype=np.float64)
        elif arr.size % 2 == 0:
            arr = arr.reshape(-1, 2)
        else:
            raise ValueError(f"Unexpected 1D longitudinal payload with odd length: shape={arr.shape}")

    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Unexpected longitudinal payload shape: {arr.shape}")

    if arr.shape[0] < target_rows:
        pad = np.full((target_rows - arr.shape[0], 2), np.nan, dtype=np.float64)
        arr = np.concatenate([arr, pad], axis=0)
    return arr.astype(np.float64, copy=False)


def _stack_longitudinal_columns(df_long_for_pred):
    target_rows = 0
    for column in LONGITUDINAL_OUTCOME_COLUMNS:
        for cell in df_long_for_pred[column].tolist():
            arr = np.asarray(cell, dtype=np.float64)
            if arr.ndim == 0:
                continue
            if arr.ndim == 1 and arr.size % 2 == 0:
                target_rows = max(target_rows, arr.size // 2)
            elif arr.ndim == 2 and arr.shape[1] == 2:
                target_rows = max(target_rows, arr.shape[0])
            else:
                raise ValueError(f"Unexpected longitudinal payload while inferring target rows: shape={arr.shape}")

    if target_rows <= 0:
        raise ValueError("Could not infer longitudinal matrix length from v5 payload.")

    stacked_columns = []
    for column in LONGITUDINAL_OUTCOME_COLUMNS:
        column_arrays = [
            _coerce_longitudinal_cell(cell, target_rows)
            for cell in df_long_for_pred[column].tolist()
        ]
        stacked_columns.append(np.stack(column_arrays, axis=0))

    return np.stack(stacked_columns, axis=1)


def import_dataset_cfdna_sim_outcome(
    norm_mode='standard',
    bspline_degree=BSPLINE_DEGREE_DEFAULT,
    bspline_time_min=BSPLINE_TIME_MIN_MONTHS,
    bspline_time_max=BSPLINE_TIME_MAX_MONTHS,
    bspline_num_internal_knots=BSPLINE_NUM_INTERNAL_KNOTS_DEFAULT,
    bspline_internal_knots=BSPLINE_INTERNAL_KNOTS_MONTHS,
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
        if col not in {"SUBJECT_NUMBER", "event_time", "event", *available_category_columns, *EXCLUDED_PREDICTOR_COLUMNS}
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

    outcome_matrix_pred_long = _stack_longitudinal_columns(df_long_for_pred)
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

    time_days = _load_time_days_for_subject_numbers(subject_numbers)
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
        time_days.astype("float32"),
    )


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
    time_days=None,
):
    tensors = [
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(E, dtype=torch.float32),
        torch.tensor(T, dtype=torch.float32),
        torch.tensor(mask1, dtype=torch.float32),
        torch.tensor(mask2, dtype=torch.float32),
        torch.tensor(outcome_matrix, dtype=torch.float32),
        torch.tensor(mask_miss, dtype=torch.float32),
        torch.tensor(mask_miss_pred, dtype=torch.float32),
        torch.tensor(batch_basis, dtype=torch.float32),
    ]
    if time_days is not None:
        tensors.append(torch.tensor(time_days, dtype=torch.float32))
    ds = TensorDataset(*tensors)
    return ds, DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


class ModelDeepHit_Multitask(base_engine.ModelDeepHit_Multitask):
    def __init__(self, input_dims, network_settings, outcome_configs, autoencoder=None, writer=None):
        super().__init__(
            input_dims=input_dims,
            network_settings=network_settings,
            outcome_configs=outcome_configs,
            autoencoder=autoencoder,
            writer=writer,
        )
        self.base_head_input_dim = int(self.h_dim_shared + self.x_dim)
        self.mask_augmented_head_input_dim = int(self.base_head_input_dim + self.x_dim)
        self.longitudinal_ipcw_eps = 1e-3
        self.longitudinal_ipcw_max_weight = 10.0
        self._longitudinal_ipcw_ref_followup_months = None

        # Survival head now sees both latent representation and prepared
        # baseline missing-mask information.
        self.cs_nets = nn.ModuleList(
            [
                base_engine.legacy.create_fc_net(
                    input_dim=self.mask_augmented_head_input_dim,
                    num_layers=self.num_layers_CS,
                    h_dim=self.h_dim_CS,
                    h_fn=self.active_fn,
                    o_dim=self.h_dim_CS,
                    o_fn=self.active_fn,
                    keep_prob=self.keep_prob,
                    use_resnet=True,
                )
                for _ in range(self.num_Event)
            ]
        )

        # Only longitudinal heads receive the augmented mask input.
        self.outcome_pred_nets = nn.ModuleList(
            [
                base_engine.legacy.create_outcome_specific_net(
                    input_dim=(
                        self.mask_augmented_head_input_dim
                        if config["task_type"] == "longitudinal_regression"
                        else self.base_head_input_dim
                    ),
                    num_layers=self.num_layers_CS,
                    hidden_dim=self.h_dim_CS,
                    activation_fn=self.active_fn,
                    output_dim=config["output_dim"],
                    output_activation=config["output_activation"],
                    keep_prob=self.keep_prob,
                    use_resnet=True,
                )
                for config in self.outcome_configs
            ]
        )

    def forward(self, x, mask):
        x_prepared, mask_prepared = self._prepare_features(x, mask)
        ae_out, _ = self.autoencoder(x_prepared, mask=mask_prepared)
        ae_out = self.linear_layer(ae_out).squeeze(-1)
        self.ae_out = ae_out

        shared_out = self.shared_net(ae_out)
        base_h = torch.cat([ae_out, shared_out], dim=1)
        mask_augmented_h = torch.cat([base_h, mask_prepared], dim=1)

        out_list = [cs_net(mask_augmented_h) for cs_net in self.cs_nets]
        out = torch.cat(out_list, dim=1)
        out = torch.nn.functional.dropout(out, p=1 - self.keep_prob, training=self.training)
        out = self.output_layer(out)
        out = torch.softmax(out, dim=1)
        out = out.view(-1, self.num_Event, self.num_Category)
        self.out = out

        outcome_preds = []
        for config, net in zip(self.outcome_configs, self.outcome_pred_nets):
            head_input = mask_augmented_h if config["task_type"] == "longitudinal_regression" else base_h
            outcome_preds.append(net(head_input))

        return self.out, outcome_preds, self.ae_out

    @staticmethod
    def _split_batch(batch):
        if len(batch) == 10:
            return batch[:9]
        if len(batch) != 9:
            raise ValueError(f"Unexpected batch length: {len(batch)}")
        return batch

    def _longitudinal_task_specs(self, outcome_configs):
        specs = []
        for head_idx, config in enumerate(outcome_configs):
            if config["task_type"] == "longitudinal_regression":
                specs.append((head_idx, int(config.get("outcome_idx", head_idx)), config))
        return specs

    def _compute_longitudinal_pretrain_loss(self, outcome_preds, outcomes_true, missing_mask_fp, basis_mb, outcome_configs):
        longitudinal_specs = self._longitudinal_task_specs(outcome_configs)
        if not longitudinal_specs:
            return None

        losses = []
        for head_idx, outcome_idx, _ in longitudinal_specs:
            losses.append(
                self.mse_longitudinal(
                    outcomes_true[:, outcome_idx, :, :],
                    outcome_preds[head_idx],
                    missing_mask_fp[:, outcome_idx, :, 0],
                    basis_mb[:, outcome_idx, :, :],
                )
            )
        return torch.mean(torch.stack(losses))

    def _evaluate_longitudinal_pretrain(self, val_loader, outcome_configs, device):
        longitudinal_specs = self._longitudinal_task_specs(outcome_configs)
        if not longitudinal_specs:
            return None

        self.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = [b.to(device) for b in batch]
                (
                    x_mb,
                    _k_mb,
                    _t_mb,
                    _m1_mb,
                    _m2_mb,
                    outcomes_true,
                    missing_mask,
                    missing_mask_fp,
                    basis_mb,
                ) = self._split_batch(batch)
                x_mb = torch.where(torch.isnan(x_mb), torch.zeros_like(x_mb), x_mb)
                _, outcome_preds, _ = self.forward(x_mb, missing_mask)
                loss = self._compute_longitudinal_pretrain_loss(
                    outcome_preds,
                    outcomes_true,
                    missing_mask_fp,
                    basis_mb,
                    outcome_configs,
                )
                if loss is None:
                    return None
                total_loss += float(loss.item())
                n_batches += 1
        return total_loss / max(n_batches, 1)

    def pretrain_longitudinal(
        self,
        train_loader,
        val_loader,
        outcome_configs,
        epochs=100,
        lr=1e-3,
        patience=10,
        device="cuda",
        save_path=None,
    ):
        longitudinal_specs = self._longitudinal_task_specs(outcome_configs)
        if not longitudinal_specs:
            print("[Longitudinal pretrain] No longitudinal task found. Skip.")
            return

        device = torch.device(device)
        self.to(device)
        ensure_checkpoint_dirs()
        if save_path is None:
            save_path = DEFAULT_LONGITUDINAL_PRETRAIN_CHECKPOINT_PATH
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        param_groups = []
        param_groups.extend(list(self.cat_embeddings.parameters()))
        param_groups.extend(list(self.autoencoder.parameters()))
        param_groups.extend(list(self.linear_layer.parameters()))
        param_groups.extend(list(self.shared_net.parameters()))
        for head_idx, _, _ in longitudinal_specs:
            param_groups.extend(list(self.outcome_pred_nets[head_idx].parameters()))

        unique_params = []
        seen = set()
        for param in param_groups:
            if param.requires_grad and id(param) not in seen:
                seen.add(id(param))
                unique_params.append(param)

        optimizer = torch.optim.Adam(unique_params, lr=lr)
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            self.train()
            total_train_loss = 0.0
            n_batches = 0
            for batch in train_loader:
                batch = [b.to(device) for b in batch]
                (
                    x_mb,
                    _k_mb,
                    _t_mb,
                    _m1_mb,
                    _m2_mb,
                    outcomes_true,
                    missing_mask,
                    missing_mask_fp,
                    basis_mb,
                ) = self._split_batch(batch)
                x_mb = torch.where(torch.isnan(x_mb), torch.zeros_like(x_mb), x_mb)

                optimizer.zero_grad()
                _, outcome_preds, _ = self.forward(x_mb, missing_mask)
                loss = self._compute_longitudinal_pretrain_loss(
                    outcome_preds,
                    outcomes_true,
                    missing_mask_fp,
                    basis_mb,
                    outcome_configs,
                )
                if loss is None:
                    print("[Longitudinal pretrain] No longitudinal loss available. Skip.")
                    return
                loss.backward()
                optimizer.step()
                total_train_loss += float(loss.item())
                n_batches += 1

            avg_train_loss = total_train_loss / max(n_batches, 1)
            val_loss = self._evaluate_longitudinal_pretrain(val_loader, outcome_configs, device)
            if val_loss is None:
                print("[Longitudinal pretrain] No validation longitudinal loss available. Skip.")
                return

            print(
                f"[Longitudinal pretrain] Epoch [{epoch + 1}/{epochs}] "
                f"Train={avg_train_loss:.6f} Val={val_loss:.6f}"
            )
            if val_loss + 1e-8 < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.state_dict(), save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[Longitudinal pretrain] Early stopping at epoch {epoch + 1}.")
                    if os.path.exists(save_path):
                        self.load_state_dict(torch.load(save_path, map_location=device))
                    break

    def set_longitudinal_ipcw_reference(self, time_days):
        time_days = time_days.detach().view(-1).float()
        self._longitudinal_ipcw_ref_followup_months = (time_days / DAYS_PER_MONTH).cpu()

    def _longitudinal_ipcw_weights(self, outcome_true_time, missing_mask_fp):
        if self._longitudinal_ipcw_ref_followup_months is None:
            return torch.ones_like(outcome_true_time)

        ref_followup_months = self._longitudinal_ipcw_ref_followup_months.to(
            device=outcome_true_time.device,
            dtype=outcome_true_time.dtype,
        )
        flat_time = outcome_true_time.reshape(-1)
        support_prob = (ref_followup_months.unsqueeze(0) >= flat_time.unsqueeze(1)).float().mean(dim=1)
        weights = (1.0 / support_prob.clamp(min=self.longitudinal_ipcw_eps)).clamp_max(
            self.longitudinal_ipcw_max_weight
        )
        weights = weights.view_as(outcome_true_time)
        observed = missing_mask_fp > 0
        if torch.any(observed):
            weights = weights / torch.clamp(weights[observed].mean(), min=self.longitudinal_ipcw_eps)
        return weights

    def mse_longitudinal(self, outcome_true, outcome_pred, missing_mask_fp, basis):
        outcome_true_time = outcome_true[:, :, 1]
        outcome_true_values = outcome_true[:, :, 0]
        outcome_true_values = torch.where(
            torch.isnan(outcome_true_values),
            torch.zeros_like(outcome_true_values),
            outcome_true_values,
        )
        outcome_true_time = torch.where(
            torch.isnan(outcome_true_time),
            torch.zeros_like(outcome_true_time),
            outcome_true_time,
        )

        outcome_pred = outcome_pred.unsqueeze(1)
        curve_prediction = torch.matmul(outcome_pred, basis).squeeze(1)
        ipcw_weight = self._longitudinal_ipcw_weights(outcome_true_time, missing_mask_fp)
        loss_elementwise = ipcw_weight * (outcome_true_values - curve_prediction) ** 2
        denom = torch.clamp((ipcw_weight * missing_mask_fp).sum(), min=1.0)
        return (loss_elementwise * missing_mask_fp).sum() / denom

    def train_model(self, train_loader, val_loader, optimizer, alpha, beta, gamma, delta, eta, fi, outcome_configs, **kwargs):
        train_batch = train_loader.dataset[:]
        raw_time_days = kwargs.pop("longitudinal_time_days", None)
        pretrain_longitudinal = bool(kwargs.pop("pretrain_longitudinal", False))
        longitudinal_pretrain_epochs = int(kwargs.pop("longitudinal_pretrain_epochs", 100))
        longitudinal_pretrain_lr = float(kwargs.pop("longitudinal_pretrain_lr", 1e-3))
        longitudinal_pretrain_patience = int(kwargs.pop("longitudinal_pretrain_patience", 10))
        longitudinal_pretrain_save_path = kwargs.pop("longitudinal_pretrain_save_path", None)
        pretrain_ae = bool(kwargs.pop("pretrain_ae", False))
        ae_pretrain_epochs = int(kwargs.pop("ae_pretrain_epochs", 50))
        ae_pretrain_lr = float(kwargs.pop("ae_pretrain_lr", 1e-3))
        ae_pretrain_sparse_weight = float(kwargs.pop("ae_pretrain_sparse_weight", 1e-4))
        ae_pretrain_patience = int(kwargs.pop("ae_pretrain_patience", 10))
        ae_pretrain_save_path = kwargs.pop("ae_pretrain_save_path", None)
        if raw_time_days is None:
            if len(train_batch) >= 10:
                raw_time_days = train_batch[9]
            else:
                raise ValueError(
                    "maskhead longitudinal IPCW requires raw event_time_days. "
                    "Use this module's import_dataset_cfdna_sim_outcome/build_tensor_dataset_and_loader "
                    "or pass longitudinal_time_days=... to train_model."
                )
        self.set_longitudinal_ipcw_reference(raw_time_days)
        device = kwargs.get("device", "cuda")

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

        if pretrain_longitudinal:
            self.pretrain_longitudinal(
                train_loader=train_loader,
                val_loader=val_loader,
                outcome_configs=outcome_configs,
                epochs=longitudinal_pretrain_epochs,
                lr=longitudinal_pretrain_lr,
                patience=longitudinal_pretrain_patience,
                device=device,
                save_path=longitudinal_pretrain_save_path,
            )

        return super().train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            eta=eta,
            fi=fi,
            outcome_configs=outcome_configs,
            pretrain_ae=False,
            **kwargs,
        )
