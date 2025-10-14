#!/bin/bash

weather_types=("rain" "snow" "fog" "rain_wet_ground" "snow_wet_ground")
noise_types=("uniform_noise" "gaussian_noise" "background_noise" "impulse_noise" \
             "uniform_noise_radial" "gaussian_noise_radial" "impulse_noise_radial" \
             "upsampling" )
density_types=("cutout" "density_inc" "density_dec" "beam_del" "layer_del")

corruptions=("${weather_types[@]}" "${noise_types[@]}" "${density_types[@]}")

corruptions=("rain" "snow" "fog" "upsampling" "uniform_noise" "gaussian_noise" "impulse_noise" "density_dec" "cutout" "layer_del")


dataset=$1
dataroot=$2
shift 2
subsets=("$@")

for corruption in "${corruptions[@]}"; do
    for severity in 1 3 5; do
        new_dataset="${dataset}_${corruption}_${severity}"
        for subset in "${subsets[@]}"; do
            cmd="cp $dataroot/$dataset/dataset/sequences/$subset/calib.txt $dataroot/$new_dataset/dataset/sequences/$subset/"
            echo $cmd
            eval $cmd
            cmd="cp $dataroot/$dataset/dataset/sequences/$subset/times.txt $dataroot/$new_dataset/dataset/sequences/$subset/"
            echo $cmd
            eval $cmd
        done
        cmd="cp -r $dataroot/$dataset/dataset/poses $dataroot/$new_dataset/dataset/"
        echo $cmd
        eval $cmd
    done
done
