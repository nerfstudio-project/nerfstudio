#!/bin/bash

for i in $(find pyrad/ -name '*.py');
do
  if ! grep -q Copyright $i
  then
    cat scripts/licensing/copyright.txt $i >$i.new && mv $i.new $i
  fi
done