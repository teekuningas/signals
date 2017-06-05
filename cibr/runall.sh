#!/bin/bash

# this script can be used to run python-script for multiple files
# with syntax:
# ./runall.sh script.py file1 file2

for fname in "${@:2}"
do
  python $1 $fname

  if [ $? -ne 0 ]
  then
	  break
  fi
done
