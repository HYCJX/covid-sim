#!/bin/sh

# shellcheck disable=SC2039
# shellcheck disable=SC2034

for i in {1..10}; do
  mkdir t"$i"
  for ((n=0;n<9;n++))
  do
    mkdir outputs
    ./run_sample.py Estonia --threads $i --outputdir outputs
    rm -rf outputs
  done
  mkdir outputs
  ./run_sample.py Estonia --threads $i --outputdir outputs
  mv outputs t"$i"
  mv Timing.txt t"$i"
done