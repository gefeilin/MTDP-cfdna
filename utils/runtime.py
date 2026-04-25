from __future__ import annotations

import sys
import types

from .config import CFDNA_ROOT_DIR, PROJECT_DIR


def install_runtime_shims() -> None:
    import numpy
    import numpy.core

    sys.modules.setdefault("numpy._core", numpy.core)

    if "sksurv.metrics" not in sys.modules:
        metrics = types.ModuleType("sksurv.metrics")

        def concordance_index_censored(*_args, **_kwargs):
            raise RuntimeError(
                "concordance_index_censored is not available in the app runtime shim. "
                "This app only uses inference-time model code."
            )

        metrics.concordance_index_censored = concordance_index_censored
        sksurv = types.ModuleType("sksurv")
        sksurv.metrics = metrics
        sys.modules["sksurv"] = sksurv
        sys.modules["sksurv.metrics"] = metrics


def add_project_paths() -> None:
    candidate_paths = [
        str(PROJECT_DIR),
        str(PROJECT_DIR / "engines"),
        str(CFDNA_ROOT_DIR),
    ]
    for path in candidate_paths:
        if path not in sys.path:
            sys.path.insert(0, path)
