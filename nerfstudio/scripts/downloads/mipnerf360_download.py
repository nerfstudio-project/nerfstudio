# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Mip-NeRF 360 dataset downloader.

Downloads the Mip-NeRF 360 scenes from an anonymously-readable bucket using the
Amazon S3 API. The endpoint and bucket are plumbing: point
``AWS_S3_ENDPOINT_URL`` (or ``--bucket``) at Amazon S3 or any S3-compatible
object store (for example Backblaze B2, Cloudflare R2, or MinIO) and the same
code path works.

Registered as the ``mipnerf360`` subcommand under ``ns-download-data``.
"""

import os
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from nerfstudio.scripts.downloads.utils import DatasetDownload
from nerfstudio.utils.rich_utils import CONSOLE

try:
    import boto3
    from botocore import UNSIGNED
    from botocore.client import Config
except ImportError:
    print("boto3 is required for the Mip-NeRF 360 downloader. Install it with `pip install boto3`.")
    sys.exit(1)


# Placeholder S3 endpoint. Override with the standard AWS_S3_ENDPOINT_URL env
# var to target Amazon S3 or any S3-compatible host.
DEFAULT_S3_ENDPOINT_URL = "https://your-s3-endpoint.example.com"

# Placeholder bucket name. Replace with the real public bucket once provisioned.
DEFAULT_BUCKET = "nerfstudio-mipnerf360"

# Mip-NeRF 360 scene names, matching the canonical Barron et al. release.
MIPNERF360_SCENES = (
    "bicycle",
    "bonsai",
    "counter",
    "garden",
    "kitchen",
    "room",
    "stump",
    "flowers",
    "treehill",
)


def _nerfstudio_user_agent_extra() -> str:
    """Return the ``nerfstudio/<version>`` user agent fragment.

    Falls back to ``nerfstudio/dev`` when the package metadata is unavailable,
    for example when running from a source checkout that was never installed.
    """
    try:
        return f"nerfstudio/{version('nerfstudio')}"
    except PackageNotFoundError:
        return "nerfstudio/dev"


def _resolve_endpoint_url() -> str:
    """Pick the S3 endpoint URL.

    Precedence: ``AWS_S3_ENDPOINT_URL`` env var, otherwise the placeholder
    default.
    """
    return os.environ.get("AWS_S3_ENDPOINT_URL", DEFAULT_S3_ENDPOINT_URL)


def _build_s3_client(anonymous: bool = True):
    """Build a boto3 S3 client for the configured S3-compatible endpoint.

    Args:
        anonymous: If True (the default for public dataset downloads), uses
            unsigned requests so users do not need credentials.

    Returns:
        A configured ``boto3.client('s3')`` instance.
    """
    config = Config(
        user_agent_extra=_nerfstudio_user_agent_extra(),
        # Virtual-hosted-style addressing works across Amazon S3 and S3-compatible stores.
        s3={"addressing_style": "virtual"},
        signature_version=UNSIGNED if anonymous else None,
    )
    return boto3.client("s3", endpoint_url=_resolve_endpoint_url(), config=config)


def _download_prefix(client, bucket: str, prefix: str, output_dir: Path) -> int:
    """Download every object under ``prefix`` from ``bucket`` into ``output_dir``.

    Returns the number of files downloaded.
    """
    paginator = client.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            key = obj["Key"]
            # Strip the prefix so the on-disk layout mirrors the scene contents,
            # not the bucket nesting.
            rel = key[len(prefix) :].lstrip("/")
            if not rel:
                continue
            dest = output_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(dest))
            count += 1
    return count


@dataclass
class Mipnerf360Download(DatasetDownload):
    """Download the Mip-NeRF 360 dataset over the Amazon S3 API.

    Uses anonymous (unsigned) reads by default. Honors ``AWS_S3_ENDPOINT_URL``
    if set, so the same command works against Amazon S3 or any S3-compatible
    object store (for example Backblaze B2, Cloudflare R2, or MinIO).
    """

    capture_name: str = "all"
    """Which Mip-NeRF 360 scene to fetch ("all" or one of the scene names)."""

    bucket: str = DEFAULT_BUCKET
    """Name of the bucket to download from. Override to use a different mirror."""

    anonymous: bool = True
    """Use unsigned requests (public buckets). Set to False to use S3 credentials from your env."""

    def download(self, save_dir: Path) -> None:
        """Download the requested scenes into ``save_dir / mipnerf360 / <scene>``."""
        if self.capture_name == "all":
            scenes = list(MIPNERF360_SCENES)
        elif self.capture_name in MIPNERF360_SCENES:
            scenes = [self.capture_name]
        else:
            CONSOLE.print(
                f"[bold red]Unknown capture '{self.capture_name}'. Valid choices: {list(MIPNERF360_SCENES) + ['all']}."
            )
            sys.exit(1)

        endpoint = _resolve_endpoint_url()
        CONSOLE.print(f"Downloading from endpoint [bold]{endpoint}[/bold], bucket [bold]{self.bucket}[/bold].")
        client = _build_s3_client(anonymous=self.anonymous)

        for i, scene in enumerate(scenes):
            prefix = f"{scene}/"
            output_dir = save_dir / "mipnerf360" / scene
            output_dir.mkdir(parents=True, exist_ok=True)
            CONSOLE.print(f"[mipnerf360 {i + 1: >2d}/{len(scenes)}] downloading scene '{scene}' to {output_dir}")
            n = _download_prefix(client, self.bucket, prefix, output_dir)
            CONSOLE.print(f"  done. {n} file(s) downloaded.")
