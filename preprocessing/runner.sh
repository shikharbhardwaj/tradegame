#!/bin/bash

IFS=$'\n'

set -f
trap "exit" INT

# Create directory if it does not exist
mkdir -p sampled_data

for i in $(cat < "$1"); do
    # Extract the file
    unzip $i
    # Check if extraction was successful
    if [ $? != 0 ]; then
        echo "Fail for $i"
    fi
    # Get the file name without extension
    filename=$(basename -- "$i")
    filename="${filename%.*}"

    # Process the file
    ./sample_fast.py "$filename.csv"
    # Check if sampling was successful
    if [ $? != 0 ]; then
        echo "Fail for $i"
    fi
    # Remove the extracted file
    rm "$filename.csv"
    if [ $? != 0 ]; then
        echo "Fail for $i"
    fi
done

