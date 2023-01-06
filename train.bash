#!/bin/bash

time=`date +"%Y-%m-%d-%T"`

mkdir run_$time
cp -r  *.py ./run_$time
rm -rf models/*
python my_RevGAN.py | tee ./run_$time/train_$time.log
if ! [[ -d models ]]; then mkdir ./models/; fi
cp -r models/ ./run_$time
