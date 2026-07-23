"""Shared runtime_env.yaml updater for ARL-backed Uni-Agent launchers."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

UPSTREAM_PLACEHOLDER_KEYS = {
    "VEFAAS_FUNCTION_ID",
    "VEFAAS_FUNCTION_ROUTE",
    "VEFAAS_REGION",
    "VOLCE_ACCESS_KEY",
    "VOLCE_SECRET_KEY",
    "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET",
    "WANDB_API_KEY",
}
UPSTREAM_COMMENT_NEEDLES = (
    "if you use vefaas",
    "if you use modal",
    "modal credentials",
    "modal token",
    "team workspace",
    "~/.modal",
    "weights & biases",
    "wandb key",
    "must pass the key through",
)

MANAGED_ENV_KEYS = (
    "TORCH_NCCL_AVOID_RECORD_STREAMS",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "CUDA_HOME",
    "CUDA_PATH",
    "CPATH",
    "LIBRARY_PATH",
    "LD_LIBRARY_PATH",
    "CMAKE_PREFIX_PATH",
    "NVTE_FRAMEWORK",
    "NVSHMEM_DIR",
    "GDRCOPY_HOME",
    "GDRCOPY_INCLUDE",
    "MAX_JOBS",
    "VLLM_WORKER_MULTIPROC_METHOD",
    "NCCL_CUMEM_ENABLE",
    "NCCL_NVLS_ENABLE",
    "NCCL_MNNVL_ENABLE",
    "NCCL_SOCKET_IFNAME",
    "GLOO_SOCKET_IFNAME",
    "VLLM_USE_NCCL_SYMM_MEM",
    "VLLM_USE_DEEP_GEMM",
    "GONGFENG_TOKEN",
    "ARL_API_KEY",
    "ARL_TOKEN",
    "ARL_GATEWAY_URL",
    "ARL_NAMESPACE",
    "ARL_EXPERIMENT_ID",
    "ARL_TIMEOUT",
    "ARL_STARTUP_TIMEOUT",
    "ARL_MIRROR_REGISTRY",
    "ARL_MIRROR_NAMESPACE",
    "P2A_ARL_IMAGE_OVERRIDES_JSON",
    "P2A_BONUS_MAP_DIR",
    "P2A_M_MAX",
    "P2A_TRACKING_MODE",
    "P2A_CREDIT_GRANULARITY",
    "P2A_EVAL_BONUS_MAP_DIR",
    "P2A_EVAL_NEAR_THRESHOLD",
    "P2A_EVAL_DETAILS_DIR",
    "UNI_AGENT_P2A_TRACE",
    "PATH",
    "UV_PROJECT_ENVIRONMENT",
    "UV_CACHE_DIR",
    "UV_HTTP_TIMEOUT",
    "UV_NO_BUILD_PACKAGE",
    "UV_PYTHON",
    "VIRTUAL_ENV",
    "WANDB_API_KEY",
    "WANDB_BASE_URL",
    "WANDB_DIR",
    "WANDB_ENTITY",
    "WANDB_MODE",
    "WANDB_NAME",
    "WANDB_PROJECT",
    "WANDB_RUN_GROUP",
    "WANDB_TAGS",
)
ARL_ENV_KEYS = (
    "TORCH_NCCL_AVOID_RECORD_STREAMS",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "CUDA_HOME",
    "CUDA_PATH",
    "CPATH",
    "LIBRARY_PATH",
    "LD_LIBRARY_PATH",
    "CMAKE_PREFIX_PATH",
    "NVTE_FRAMEWORK",
    "NVSHMEM_DIR",
    "GDRCOPY_HOME",
    "GDRCOPY_INCLUDE",
    "MAX_JOBS",
    "VLLM_WORKER_MULTIPROC_METHOD",
    "NCCL_CUMEM_ENABLE",
    "NCCL_NVLS_ENABLE",
    "NCCL_MNNVL_ENABLE",
    "VLLM_USE_NCCL_SYMM_MEM",
    "VLLM_USE_DEEP_GEMM",
    "ARL_GATEWAY_URL",
    "ARL_NAMESPACE",
    "ARL_EXPERIMENT_ID",
    "ARL_TIMEOUT",
    "ARL_STARTUP_TIMEOUT",
    "ARL_MIRROR_REGISTRY",
    "ARL_MIRROR_NAMESPACE",
    "P2A_ARL_IMAGE_OVERRIDES_JSON",
    "P2A_BONUS_MAP_DIR",
    "P2A_M_MAX",
    "P2A_TRACKING_MODE",
    "P2A_CREDIT_GRANULARITY",
    "P2A_EVAL_BONUS_MAP_DIR",
    "P2A_EVAL_NEAR_THRESHOLD",
    "P2A_EVAL_DETAILS_DIR",
    "UNI_AGENT_P2A_TRACE",
)
REFERENCE_ENV_DEFAULTS = {
    "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "CUDA_HOME": "/usr/local/cuda-13.0",
    "CUDA_PATH": "/usr/local/cuda-13.0",
    "NVTE_FRAMEWORK": "pytorch",
    "MAX_JOBS": "32",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "NCCL_CUMEM_ENABLE": "1",
    "NCCL_NVLS_ENABLE": "1",
    "NCCL_MNNVL_ENABLE": "1",
    "VLLM_USE_NCCL_SYMM_MEM": "1",
    "VLLM_USE_DEEP_GEMM": "0",
}


def _replace_yaml_value(text: str, key: str, value: str) -> tuple[str, bool]:
    pattern = rf"(^\s*{re.escape(key)}:\s*).*$"

    def repl(match: re.Match[str]) -> str:
        return f"{match.group(1)}{json.dumps(value)}"

    next_text, n_replaced = re.subn(pattern, repl, text, flags=re.MULTILINE)
    return next_text, n_replaced > 0


def update_runtime_env(
    path: Path,
    pythonpath: str,
    *,
    drop_working_dir: bool,
    prune_empty: bool,
    env_keys: tuple[str, ...] = MANAGED_ENV_KEYS,
) -> None:
    text = path.read_text(encoding="utf-8")
    text, found_pythonpath = _replace_yaml_value(text, "PYTHONPATH", pythonpath)

    lines: list[str] = []
    skip_top_level_block = False
    for line in text.splitlines():
        stripped = line.strip()
        if not line.startswith(" "):
            skip_top_level_block = False
            if drop_working_dir and (
                stripped.startswith("working_dir:")
                or stripped.startswith("excludes:")
                or stripped.startswith("pip:")
            ):
                skip_top_level_block = True
                continue
        elif skip_top_level_block:
            continue
        if any(needle in stripped.lower() for needle in UPSTREAM_COMMENT_NEEDLES):
            continue
        key = stripped.split(":", 1)[0]
        if key in UPSTREAM_PLACEHOLDER_KEYS:
            continue
        lines.append(line)

    text = "\n".join(lines) + "\n"
    if not found_pythonpath and "env_vars:" in text:
        text = text.rstrip() + f"\n  PYTHONPATH: {json.dumps(pythonpath)}\n"

    for key, default in REFERENCE_ENV_DEFAULTS.items():
        if key not in os.environ:
            text, replaced = _replace_yaml_value(text, key, default)
            if not replaced and "env_vars:" in text:
                text = text.rstrip() + f"\n  {key}: {json.dumps(default)}\n"

    for key in env_keys:
        value = os.environ.get(key)
        pattern = rf"(^\s*{re.escape(key)}:\s*).*$"
        if not value:
            if prune_empty and key not in REFERENCE_ENV_DEFAULTS:
                text = re.sub(pattern + r"\n?", "", text, flags=re.MULTILINE)
            continue
        text, replaced = _replace_yaml_value(text, key, value)
        if not replaced and "env_vars:" in text:
            text = text.rstrip() + f"\n  {key}: {json.dumps(value)}\n"

    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime_env", type=Path)
    parser.add_argument("--src-root", default=None, help="Use absolute src-root PYTHONPATH entries.")
    parser.add_argument("--pythonpath", default=None, help="Explicit PYTHONPATH value.")
    parser.add_argument("--drop-working-dir", action="store_true")
    parser.add_argument("--env-profile", choices=("train", "arl"), default="train")
    parser.add_argument("--preserve-empty-env", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.pythonpath:
        pythonpath = args.pythonpath
    elif args.src_root:
        src_root = str(Path(args.src_root).resolve())
        pythonpath = f"{src_root}/uni-agent/verl:{src_root}/uni-agent:{src_root}"
    else:
        pythonpath = "uni-agent/verl:uni-agent:."

    update_runtime_env(
        args.runtime_env,
        pythonpath,
        drop_working_dir=args.drop_working_dir,
        prune_empty=not args.preserve_empty_env,
        env_keys=ARL_ENV_KEYS if args.env_profile == "arl" else MANAGED_ENV_KEYS,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
