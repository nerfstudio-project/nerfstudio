# Nerfactory completions for bash
#
# This can be sourced in your .bashrc:
#
#     source ~/nerfactory/scripts/completions/setup.bash
#

completions_dir="$(dirname "$BASH_SOURCE")"/bash

if [ ! -d "${completions_dir}" ]; then
  echo "$0: Completions are missing!"
  echo "Please generate them with nerfactory/scripts/completions/generate.py!"
  return 1
fi

# Source each completion script.
for completion_path in ${completions_dir}/*
do
    source $completion_path
done
