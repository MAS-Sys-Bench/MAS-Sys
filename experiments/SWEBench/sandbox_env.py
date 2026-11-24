"""Bridge MAS-Sys sandbox creation to the embedded SWE-bench harness.

For SWE-bench generation we prepare the same instance image that the embedded
harness uses for evaluation. The image contains the repo at ``/testbed`` after
the official repo setup script has run, so agents operate in the same prepared
environment instead of trying to repair dependencies at runtime.
"""

from __future__ import annotations

from typing import Dict

import docker

from app.config import SandboxSettings
from app.logger import logger
from experiments.SWEBench.harness.docker_build import build_instance_images
from experiments.SWEBench.harness.test_spec.test_spec import LATEST
from experiments.SWEBench.harness.test_spec.test_spec import make_test_spec


def resolve_sandbox_config_for_instance(
    instance: Dict,
    repo_dir: str,
    base_config: SandboxSettings,
) -> SandboxSettings:
    """Resolve the sandbox image for one SWE-bench instance."""
    del repo_dir

    image_name = ensure_instance_image(instance)
    logger.info(
        "[SWE-bench] {}: Resolved sandbox instance image {}",
        instance.get("instance_id", "unknown"),
        image_name,
    )
    return base_config.model_copy(update={"image": image_name})


def ensure_instance_image(instance: Dict) -> str:
    """Build or reuse the SWE-bench harness instance image for one instance."""
    test_spec = make_test_spec(
        instance,
        base_image_tag=LATEST,
        env_image_tag=LATEST,
        instance_image_tag=LATEST,
    )
    image_name = test_spec.instance_image_key
    client = docker.from_env()

    try:
        client.images.get(image_name)
        return image_name
    except docker.errors.ImageNotFound:
        build_instance_images(
            client,
            [instance],
            max_workers=1,
            tag=LATEST,
            env_image_tag=LATEST,
        )

    try:
        client.images.get(image_name)
        return image_name
    except docker.errors.ImageNotFound as exc:
        raise RuntimeError(
            f"Embedded harness instance image was not created: {image_name}"
        ) from exc
