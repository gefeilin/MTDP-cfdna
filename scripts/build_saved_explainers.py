from __future__ import annotations

import argparse
import sys
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from utils.config import SHAP_BACKGROUND_SIZE, TARGET_SPECS
from utils.modeling import get_prediction_service
from utils.shap_utils import save_kernel_explainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and save cfDNA target-specific Kernel SHAP explainers.",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        default=list(TARGET_SPECS.keys()),
        help="Target keys to build. Defaults to all app targets.",
    )
    parser.add_argument(
        "--background-size",
        type=int,
        default=SHAP_BACKGROUND_SIZE,
        help="Background cohort size used to build each explainer.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite any existing .dill explainer files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    invalid = [target for target in args.targets if target not in TARGET_SPECS]
    if invalid:
        raise SystemExit(f"Unknown target key(s): {', '.join(invalid)}")

    service = get_prediction_service()
    for target_key in args.targets:
        path = save_kernel_explainer(
            service,
            target_key,
            background_size=args.background_size,
            overwrite=args.overwrite,
        )
        print(f"saved {target_key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
