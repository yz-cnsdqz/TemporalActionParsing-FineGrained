#!/bin/bash
for i in `seq 1 30`;
do
    echo "--iteration $i"
    python TCN_main.py
done
