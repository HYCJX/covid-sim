#!/bin/sh

# shellcheck disable=SC2039
# shellcheck disable=SC2034
for ((n=0;n<1;n++))
do
 mkdir outputs
 ./run_sample.py Estonia --threads 8 --outputdir outputs
 rm -rf outputs
done
 mkdir outputs
./run_sample.py Estonia --threads 8 --outputdir outputs
