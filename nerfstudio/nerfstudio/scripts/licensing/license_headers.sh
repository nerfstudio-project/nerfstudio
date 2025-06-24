#!/bin/bash

VALID_ARGS=$(getopt -o c --long check -- "$@")

eval set -- "$VALID_ARGS"
check=false
while [ : ]; do
  case "$1" in
    -c | --check)
        check=true
        shift
        ;;
    --) shift;
        break 
        ;;
  esac
done

check_failed=false
added_headers=false
for i in $(find nerfstudio/ -name '*.py');
do
  if ! grep -q Copyright $i
  then
    if [ "$check" = true ];
      then
        echo "$i missing copyright header"
        check_failed=true
      else
        cat nerfstudio/scripts/licensing/copyright.txt $i >$i.new && mv $i.new $i
        echo "Adding license header to $i."
      fi
    added_headers=true
  fi
done

if [ "$check_failed" = true ];
  then
    echo "Run '.nerfstudio/scripts/licensing/license_headers.sh to add missing headers.'"
    exit 1
fi

if [ "$added_headers" = false ];
  then
    echo "No missing license headers found."
fi

exit 0