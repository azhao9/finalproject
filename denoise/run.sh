#!/bin/bash

declare -a noise_types=("gaussian" "saltpepper" "speckle")

## now loop through the above array
for noise_type in "${noise_types[@]}"
do
   for image in raw_orig_imgs/*
   do
	echo "$image"
	python gibbs_denoise.py "$noise_type" "$image" >> "gibbs_${noise_type}_results.txt"
   done

   # or do whatever with individual element of the array
done

##noise_types = ['gaussian', 'saltpepper', 'speckle']

