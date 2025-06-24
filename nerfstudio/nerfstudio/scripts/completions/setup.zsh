# nerfstudio completions for zsh.
#
# This should generally be installed automatically by `configure.py`.

completions_dir="${0:a:h}"/zsh

if [ ! -d "${completions_dir}" ]; then
  echo "$0: Completions are missing!"
  echo "Please generate them with nerfstudio/scripts/completions/generate.py!"
  return 1
fi

# Manually load and define each completion.
#
# Adding the completions directory to our fpath and re-initializing would work
# as well:
#     fpath+=${completions_dir}
#     autoload -Uz compinit; compinit
# But would be several orders of magnitude slower.
for completion_path in ${completions_dir}/*
do
  # /some/path/to/_our_completion_py => _our_completion_py
  completion_name=${completion_path##*/}
  if [[ $name == *_py ]]; then
    # _our_completion_py => our_completion.py
    script_name="${completion_name:1:-3}.py"
  else
    # _entry-point => entry-point
    script_name="${completion_name:1}"
  fi

  autoload -Uz $completion_path
  compdef $completion_name $script_name
done
