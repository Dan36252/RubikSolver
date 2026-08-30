#!/bin/bash

PREFIX="6-"
DIRECTORY="./"

for filename in "$DIRECTORY"/*; do
  if [ -f "$filename" ]; then
    basefile=$(basename "$filename")
    mv "$filename" "$DIRECTORY/$PREFIX$basefile"
    echo "Renamed: $basefile -> $PREFIX$basefile"
  fi
done
