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

"""Unit tests for the Mip-NeRF 360 (Amazon S3 API) dataset downloader."""

from unittest import mock

import pytest


def test_resolve_endpoint_url_default(monkeypatch):
    """Default endpoint is the placeholder S3 URL."""
    from nerfstudio.scripts.downloads import mipnerf360_download

    monkeypatch.delenv("AWS_S3_ENDPOINT_URL", raising=False)
    assert mipnerf360_download._resolve_endpoint_url() == mipnerf360_download.DEFAULT_S3_ENDPOINT_URL


def test_resolve_endpoint_url_env_override(monkeypatch):
    """AWS_S3_ENDPOINT_URL takes precedence over the default."""
    from nerfstudio.scripts.downloads import mipnerf360_download

    monkeypatch.setenv("AWS_S3_ENDPOINT_URL", "https://s3.example.test")
    assert mipnerf360_download._resolve_endpoint_url() == "https://s3.example.test"


def test_build_s3_client_sets_user_agent_and_addressing(monkeypatch):
    """boto3.client is called with the nerfstudio user agent extra and virtual addressing."""
    from nerfstudio.scripts.downloads import mipnerf360_download

    fake_client = mock.MagicMock()
    with mock.patch.object(mipnerf360_download, "boto3") as fake_boto3:
        fake_boto3.client.return_value = fake_client
        result = mipnerf360_download._build_s3_client(anonymous=True)

    assert result is fake_client
    fake_boto3.client.assert_called_once()
    args, kwargs = fake_boto3.client.call_args
    assert args[0] == "s3"
    assert "endpoint_url" in kwargs
    config = kwargs["config"]
    assert config.user_agent_extra.startswith("nerfstudio/")
    assert config.s3 == {"addressing_style": "virtual"}


def test_user_agent_extra_uses_package_version(monkeypatch):
    """The user agent fragment is nerfstudio/<version> when metadata is available."""
    from nerfstudio.scripts.downloads import mipnerf360_download

    monkeypatch.setattr(mipnerf360_download, "version", lambda name: "9.9.9")
    assert mipnerf360_download._nerfstudio_user_agent_extra() == "nerfstudio/9.9.9"


def test_user_agent_extra_falls_back_to_dev(monkeypatch):
    """The user agent fragment falls back to nerfstudio/dev without package metadata."""
    from nerfstudio.scripts.downloads import mipnerf360_download

    def _raise(name):
        raise mipnerf360_download.PackageNotFoundError(name)

    monkeypatch.setattr(mipnerf360_download, "version", _raise)
    assert mipnerf360_download._nerfstudio_user_agent_extra() == "nerfstudio/dev"


def test_build_s3_client_anonymous_uses_unsigned():
    """anonymous=True selects the UNSIGNED signature version."""
    from nerfstudio.scripts.downloads import mipnerf360_download

    with mock.patch.object(mipnerf360_download, "boto3") as fake_boto3:
        mipnerf360_download._build_s3_client(anonymous=True)
    config = fake_boto3.client.call_args.kwargs["config"]
    assert config.signature_version == mipnerf360_download.UNSIGNED


def test_build_s3_client_signed_when_not_anonymous():
    """anonymous=False does not force the UNSIGNED signature version."""
    from nerfstudio.scripts.downloads import mipnerf360_download

    with mock.patch.object(mipnerf360_download, "boto3") as fake_boto3:
        mipnerf360_download._build_s3_client(anonymous=False)
    config = fake_boto3.client.call_args.kwargs["config"]
    assert config.signature_version is None


@pytest.mark.parametrize("scene", ["bicycle", "garden", "treehill"])
def test_mipnerf360_download_single_scene_invokes_client(monkeypatch, tmp_path, scene):
    """Downloading a specific scene paginates and downloads each listed object."""
    from nerfstudio.scripts.downloads import mipnerf360_download

    fake_client = mock.MagicMock()
    paginator = mock.MagicMock()
    paginator.paginate.return_value = [
        {"Contents": [{"Key": f"{scene}/images/0001.png"}, {"Key": f"{scene}/transforms.json"}]},
    ]
    fake_client.get_paginator.return_value = paginator

    monkeypatch.setattr(mipnerf360_download, "_build_s3_client", lambda anonymous=True: fake_client)

    dl = mipnerf360_download.Mipnerf360Download(capture_name=scene, bucket="nerfstudio-mipnerf360")
    dl.download(tmp_path)

    paginator.paginate.assert_called_once_with(Bucket="nerfstudio-mipnerf360", Prefix=f"{scene}/")
    assert fake_client.download_file.call_count == 2


def test_mipnerf360_download_unknown_scene_exits(monkeypatch, tmp_path):
    """An unknown capture_name calls sys.exit(1)."""
    from nerfstudio.scripts.downloads import mipnerf360_download

    dl = mipnerf360_download.Mipnerf360Download(capture_name="not-a-scene")
    with pytest.raises(SystemExit):
        dl.download(tmp_path)
