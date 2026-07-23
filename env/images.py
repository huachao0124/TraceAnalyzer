"""Image routing for R2E/Uni-Agent runs.

``scripts/build_data.py`` writes the target registry ref into each parquet
row's ``deployment.image`` at build time (controlled by ``P2A_IMAGE_REGISTRY``).
This module selects the image for a given instance, with optional per-instance
overrides via ``P2A_ARL_IMAGE_OVERRIDES_JSON``.
"""

from __future__ import annotations

import json
import os
from typing import Any


def repo_from_instance_id(instance_id: str) -> str | None:
    if "__" not in instance_id:
        return None
    repo = instance_id.rsplit("__", 1)[0].strip().lower()
    return repo or None


def suffix_from_instance_id(instance_id: str) -> str | None:
    if "__" not in instance_id:
        return None
    suffix = instance_id.rsplit("__", 1)[1].strip().lower()
    return suffix or None


def _env_image_overrides() -> dict[str, str]:
    raw = os.getenv("P2A_ARL_IMAGE_OVERRIDES_JSON", "")
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("P2A_ARL_IMAGE_OVERRIDES_JSON must be a JSON object") from exc
    if not isinstance(data, dict):
        raise ValueError("P2A_ARL_IMAGE_OVERRIDES_JSON must be a JSON object")
    return {str(k): str(v) for k, v in data.items()}


def select_r2e_image(
    *,
    instance_id: str,
    docker_image: str | None = None,
) -> str:
    """Select the image for an R2E instance.

    Resolution order:
      1. exact per-instance override (``P2A_ARL_IMAGE_OVERRIDES_JSON``);
      2. the ``docker_image`` carried by the parquet row (written by
         ``scripts/build_data.py`` with the registry set by ``P2A_IMAGE_REGISTRY``).
    """

    overrides = _env_image_overrides()
    if instance_id in overrides:
        return overrides[instance_id]

    if docker_image:
        return docker_image

    raise ValueError(
        f"Cannot select image for {instance_id!r}: no docker_image in the parquet row. "
        f"Rebuild with: python scripts/build_data.py r2e"
    )


def select_image_for_sample(sample_or_task: dict[str, Any], *, instance_id: str | None = None) -> str:
    """Select an image from a Uni-Agent sample or raw R2E task dict."""

    extra = sample_or_task.get("extra_info")
    if isinstance(extra, str):
        try:
            extra = json.loads(extra)
        except json.JSONDecodeError:
            extra = {}
    if not isinstance(extra, dict):
        extra = {}

    tools_kwargs = extra.get("tools_kwargs") if isinstance(extra.get("tools_kwargs"), dict) else {}
    env = tools_kwargs.get("env") if isinstance(tools_kwargs.get("env"), dict) else {}
    deployment = env.get("deployment") if isinstance(env.get("deployment"), dict) else {}
    reward = tools_kwargs.get("reward") if isinstance(tools_kwargs.get("reward"), dict) else {}
    metadata = reward.get("metadata") if isinstance(reward.get("metadata"), dict) else {}

    iid = (
        instance_id
        or sample_or_task.get("instance_id")
        or metadata.get("instance_id")
        or (f"{sample_or_task.get('repo_name')}__{str(sample_or_task.get('commit_hash', ''))[:10]}"
            if sample_or_task.get("repo_name") and sample_or_task.get("commit_hash")
            else None)
    )
    if not iid:
        raise ValueError("Cannot select image: sample has no instance_id")

    docker_image = deployment.get("image") or env.get("image") or sample_or_task.get("docker_image")
    return select_r2e_image(instance_id=str(iid), docker_image=docker_image)
