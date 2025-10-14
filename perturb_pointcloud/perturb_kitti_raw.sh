#!/bin/bash

weather_types=("rain" "snow" "fog" "rain_wet_ground" "snow_wet_ground")
noise_types=("uniform_noise" "gaussian_noise" "background_noise" "impulse_noise" \
             "uniform_noise_radial" "gaussian_noise_radial" "impulse_noise_radial" \
             "upsampling" )
density_types=("cutout" "density_inc" "density_dec" "beam_del" "layer_del")

corruptions=("${weather_types[@]}" "${noise_types[@]}" "${density_types[@]}")

dataset=$1
dataroot=$2
nworkers=$3
subsets=( "2011_09_30/2011_09_30_drive_0018_extract" "2011_09_30/2011_09_30_drive_0020_extract" )

base_cmd="python perturb_pointcloud.py --dataset $dataset --dataroot $dataroot --nworkers $nworkers"

for subset in "${subsets[@]}"; do
    for corruption in "${corruptions[@]}"; do
        for severity in 1 3 5; do
            cmd="$base_cmd --subset $subset --corruption $corruption --severity $severity"
            echo $cmd
            eval $cmd
        done
    done
done
