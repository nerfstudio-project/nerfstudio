#!/bin/bash -e

show_help() {
  echo -e "\nUsage: $0 [OPTIONS] <commands>\n"
  echo "Options:"
  echo "  --download-src    The source file or folder to download"
  echo "  --download-dest   The destination file or folder to download to"
  echo "  --upload-src      The source file or folder to upload"
  echo "  --upload-dest     The destination file or folder to upload to"
  echo -e "\nThis script downloads the necessary files, executes the specified commands, and then uploads the output files.\n"
}

resolve_path() {
  local path="$1"
  if [[ "$path" != "omniverse://"* && ! "$path" == /* ]]; then
    path="$(pwd)/$path"
  fi
  echo "$path"
}

# Ref: https://stackoverflow.com/a/14203146
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --download-src)
      DOWNLOAD_SRC=$(resolve_path "$2")
      shift # past argument
      shift # past value
      ;;
    --download-dest)
      DOWNLOAD_DEST=$(resolve_path "$2")
      shift # past argument
      shift # past value
      ;;
    --upload-src)
      UPLOAD_SRC=$(resolve_path "$2")
      shift # past argument
      shift # past value
      ;;
    --upload-dest)
      UPLOAD_DEST=$(resolve_path "$2")
      shift # past argument
      shift # past value
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters

if [ "$#" -lt 1 ]; then
  echo "Error: Incorrect number of arguments. Expected more than 1, got $#."
  show_help
  exit 1
fi

if [ -n "$DOWNLOAD_SRC" ] || [ -n "$DOWNLOAD_DEST" ]; then
  if [ -e "$DOWNLOAD_DEST" ]; then
    echo "File exists at '$DOWNLOAD_DEST', deleting..."
    rm -rf "$DOWNLOAD_DEST"
  fi
  echo "Copying files from '$DOWNLOAD_SRC' to '$DOWNLOAD_DEST'..."
  ( cd /omnicli && ./omnicli copy "$DOWNLOAD_SRC" "$DOWNLOAD_DEST" )
fi

echo "Will run commands: '$@'"
while [[ $# -gt 0 ]]; do
  echo "Running command: '$1'"
  $1
  shift
done

if [ -n "$UPLOAD_SRC" ] || [ -n "$UPLOAD_DEST" ]; then
  echo "Copying files from '$UPLOAD_SRC' to '$UPLOAD_DEST'..."
  ( cd /omnicli && ./omnicli copy "$UPLOAD_SRC" "$UPLOAD_DEST" )
fi
