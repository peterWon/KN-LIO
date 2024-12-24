echo Runing PIN-LIO on HILTI22

for seq in exp04_construction_upper_level exp05_construction_upper_level_2 exp06_construction_upper_level_3 exp14_basement_2 exp16_attic_to_upper_gallery_2 exp18_corridor_lower_gallery_2; do
    python3 pin_slam.py config/hilti22/${seq}.yaml
    echo Processed ${seq}.
done