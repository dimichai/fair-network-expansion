#!/bin/bash

if [ ${#@} -gt 1 ]
then
    echo "Only one result_dir allowed (for now)"
    exit 1
fi

if [[ -d $1 ]]
then
    # Remove the checkpoint path, result_path and test flags
    args=$(cat "$1/args.txt" | grep -v -e checkpoint -e result_path -e test)
    # Remove special characters
    args=$(echo $args | sed 's/[":{}]//g')
    # Replace , with -- (and add a comma to the beginning of args)
    args=$(echo ,$args | sed 's/, / --/g')
    # Add back the result flag and test flag
    args="$args --result_path $1 --test"
    eval "python main.py $args"
else
    echo "Result dir not found!"
    exit 1
fi
