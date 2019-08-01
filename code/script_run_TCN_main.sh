#!/bin/bash
pooling=$1
dataset=$2
for i in 8;
do
    for j in 8 16;
    do
	out_file_name=result_rebuttal_"$dataset"_"$pooling"_dim"$i"_comp"$j"_drp0.4_ep500.txt
        python3 TCN_main.py $pooling $i 500 0.0001 8 $j $dataset $out_file_name
	echo dataset is $dataset, pooling is $pooling, dimension is $i, n_components is $j
    done
done








