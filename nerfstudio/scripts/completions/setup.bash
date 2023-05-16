# nerfstudio completions for bash.
#
# This should generally be installed automatically by `configure.py`.

completions_dir="$(dirname "$BASH_SOURCE")"/bash

if [ ! -d "${completions_dir}" ]; then
  echo "$0: Completions are missing!"
  echo "Please generate them with nerfstudio/scripts/completions/generate.py!"
  return 1
fi

# Source each completion script.
for completion_path in ${completions_dir}/*
do
    source $completion_path
done
