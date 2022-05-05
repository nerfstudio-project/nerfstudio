#!/bin/bash

helpFunction_flame()
{
   echo "Usage: $0 -o output_file -p program"
   echo -e "\t-o output .svg filename, where to save flame graph"
   echo -e "\t-p .py program we want to profile with associated arguments"
   exit 1 # Exit program after printing help
}

helpFunction_top()
{
   echo "Usage: $0 -p program"
   echo -e "\t-p .py program we want to profile with associated arguments"
   exit 1 # Exit program after printing help
}


while getopts "t:o:p:a:" opt; do
    case "$opt" in
        t ) type="$OPTARG" ;;
        o ) output_file="$OPTARG" ;;
        p ) program="$OPTARG" ;;
        ? ) helpFunction ;; 
    esac
    done
    if [ -z "$type" ]; then
        echo "Missing profiling type (flame / top)"
    fi
    # Print helpFunction in case parameters are empty
    if [[ "$type" = "flame" ]]; then
        if [ -z "$output_file" ] || [ -z "$program" ]; then
            echo "Some or all of the parameters are empty";
            helpFunction_flame
        fi
        py-spy record -o "$output_file" python "$program" 
    elif [[ "$type" = "top" ]]; then
        if [ -z "$program" ]; then
            echo "Some or all of the parameters are empty";
            helpFunction_top
        fi
        py-spy top python "$program" 
    fi