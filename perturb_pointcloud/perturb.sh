#!/bin/bash

weather_types=("rain" "snow" "fog" "rain_wet_ground" "snow_wet_ground")
noise_types=("uniform_noise" "gaussian_noise" "background_noise" "impulse_noise" \
             "uniform_noise_radial" "gaussian_noise_radial" "impulse_noise_radial" \
             "upsampling" )
density_types=("cutout" "density_inc" "density_dec" "beam_del" "layer_del")

corruptions=("${weather_types[@]}" "${noise_types[@]}" "${density_types[@]}")

corruptions=("rain_wet_ground" "snow_wet_ground")

dataset=$1
dataroot=$2
nworkers=$3
shift 3
subsets=("$@")

base_cmd="python perturb_pointcloud.py --dataset $dataset --dataroot $dataroot --nworkers $nworkers"

for subset in "${subsets[@]}"; do
    for corruption in "${corruptions[@]}"; do
        for severity in {1..5}; do
            cmd="$base_cmd --subset $subset --corruption $corruption --severity $severity"
            echo $cmd
            eval $cmd
        done
    done
done
