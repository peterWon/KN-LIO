# Dependencies

1. `conda create -n perturb_pc python=3.8 scikit-learn numpy scipy click matplotlib`
2. `conda activate perturb_pc`
3. `pip install pypcd4 multiprocess`
4. `git clone --depth 1 https://github.com/velatkilic/LISA.git && cd LISA && python setup.py install`
5. copy pylisa/mie.so to where pylisa is installed (likely to be ~/miniconda3/envs/perturb_pc/lib/python3.8/site-packages/pylisa-0.0.2-py3.8.egg/pylisa/)

# Datasets
Please find the clean data from [KITTI Odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) and [Apollo Southbay](https://developer.apollo.auto/southbay.html). 
For KITTI, you will only need to download velodyne laser data, calibration files and ground truth poses. 
For Apollo Southbay, you will only need to download ColumbiaPark. 
You should keep the original folder structure. 

# Acknowledgement

1. [density_corruption.py](./density_corruption.py) and [noise_corruption.py](./density_corruption.py) are from https://github.com/Castiel-Lee/robustness_pc_detector/tree/main with some modifications.
2. [fog_simulation.py](./fog_simulation.py) as well as [integral_lookup_tables](./integral_lookup_tables/) is from https://github.com/MartinHahner/LiDAR_fog_sim/tree/main.
3. [wet_ground](./wet_ground/) is from https://github.com/SysCV/LiDAR_snow_sim/tree/main.
4. LISA is installed as a dependency from https://github.com/velatkilic/LISA/tree/main.

We would like thank for their great work!
