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

"""
Code for deploying the built viewer folder to a server and handing versioning.
We use the library sshconf (https://github.com/sorend/sshconf) for working with the ssh config file.
"""
import json
import subprocess
from os.path import expanduser
from pathlib import Path
from typing import Optional

import tyro
from sshconf import empty_ssh_config_file, read_ssh_config


def run_cmd(cmd: str):
    """Run a command in the terminal."""
    print("cmd:", cmd)
    print("output:")
    subprocess.Popen(cmd, shell=True).wait()


def main(
    branch_name: str = "",
    ssh_key_string: Optional[str] = None,
    ssh_key_filename: str = " ~/.ssh/github_actions_user_key_filename",
    hostname_or_ip_address: str = "34.102.68.79",
    local_folder: str = "/home/eweb0124/build",
    remote_folder: str = "/home/eweb0124/viewer",
    host: str = "viewer_deploy_host",
    user: str = "eweb0124",
    package_json_filename: str = "package.json",
    increment_version: str = "False",
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
    print("package_json_filename", package_json_filename)
    print("increment_version", increment_version)
    print()

    # save the ssh key to a file
    run_cmd("mkdir ~/.ssh")
    if ssh_key_string:
        run_cmd(f"""rm -f {ssh_key_filename}""")
        run_cmd(f"""echo "{ssh_key_string}" >> {ssh_key_filename}""")
        run_cmd(f"chmod 400 {ssh_key_filename}")

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

    # show that the config is correct
    run_cmd("cat ~/.ssh/config")

    # save the config file
    config.save()

    # get the version
    with open(package_json_filename, "r", encoding="utf-8") as f:
        package_json = json.load(f)

    # TODO: add logic to increment the version number
    if increment_version == "True":
        raise NotImplementedError()
    else:
        version = package_json["version"]

    print(f"\nusing version: {version}")

    # write to the /home/eweb0124/build folder
    run_cmd(f"""ssh {host} 'rm -rf /home/eweb0124/build'""")
    run_cmd(f"""scp -r {local_folder} {host}:/home/eweb0124/build""")

    # update the symlink of latest
    if branch_name == "main":
        # move the build folder to the correct location
        run_cmd(f"""ssh {host} 'rm -rf {remote_folder}/{version}'""")
        run_cmd(f"""ssh {host} 'cp -R /home/eweb0124/build {remote_folder}/{version}'""")

        run_cmd(f"""ssh {host} 'rm {remote_folder}/latest'""")
        run_cmd(f"""ssh {host} 'ln -s {remote_folder}/{version} {remote_folder}/latest'""")

    # otherwise just move to some branch folder
    else:
        updated_branch_name = branch_name.replace("/", "-")
        run_cmd(f"""ssh {host} 'rm -rf {remote_folder}/branch/{updated_branch_name}'""")
        run_cmd(f"""ssh {host} 'cp -R /home/eweb0124/build {remote_folder}/branch/{updated_branch_name}'""")


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(main)
