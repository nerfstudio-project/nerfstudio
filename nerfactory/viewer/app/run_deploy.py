# Copyright 2022 The Plenoptix Team. All rights reserved.
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

"""
Code for handling the version of the application.
We use the library sshconf (https://github.com/sorend/sshconf) for working with the ssh config file.
"""
from pathlib import Path
import os

from typing import Optional
from os.path import expanduser

import dcargs
from sshconf import empty_ssh_config_file, read_ssh_config


def get_version(old_version: Optional[str] = None):
    return "22-09-06-1"


def main(
    branch_name: str = "",
    ssh_key_string: str = "",
    ssh_key_filename: str = " ~/.ssh/github_actions_user_key_filename",
    hostname_or_ip_address: str = "34.102.68.79",
    local_folder: str = "temp",
    remote_folder: str = "temp",
    host: str = "viewer_deploy_host",
    user: str = "eweb0124",
):
    """Copy a local folder to a remote machine and handle versioning.

    Args:
        ssh_key: The private ssh key needed to ssh.
        hostname_or_ip_address: The hostname or ip_address of the remote machine.
    """

    print()
    print("branch_name", branch_name)
    print("ssh_key_string", ssh_key_string)
    print("ssh_key_filename", ssh_key_filename)
    print("hostname_or_ip_address", hostname_or_ip_address)
    print("local_folder", local_folder)
    print("remote_folder", remote_folder)
    print("host", host)
    print("user", user)
    print()

    # save the ssh key to a file
    os.system("mkdir ~/.ssh")
    os.system(f"echo {ssh_key_string} >> {ssh_key_filename}")
    os.system(f"chmod 400 {ssh_key_filename}")

    # setup the config in ~/.ssh/config
    config_filename = expanduser("~/.ssh/config")
    Path(config_filename).parent.mkdir(exist_ok=True)
    try:
        config = read_ssh_config(config_filename)
    except FileNotFoundError:
        config = empty_ssh_config_file()
        config.write(config_filename)
        config = read_ssh_config(config_filename)

    # add the host if it doesn't exist
    if not config.host(host):
        config.add(host)

    config.set(
        host,
        Hostname=hostname_or_ip_address,
        User=user,
        IdentityFile=ssh_key_filename,
        StrictHostKeyChecking="No",
    )

    # save the config file
    config.save()

    # get the version of master
    version_master = "22-09-2021-0"
    version_new = get_version(version_master)

    target_path = "/path/to/folder/{version_new}"
    symlink_path = "/path/to/folder/latest"

    os.system("""ssh github_action_remote_machine 'rm -rf /home/eweb0124/build'""")
    os.system("""ssh github_action_remote_machine 'mkdir /home/eweb0124/viewer/branch/temp'""")

    print("target path")
    print(target_path)


if __name__ == "__main__":
    dcargs.cli(main)
