<p align="center">

<h1 align="center">KN-LIO: Kinematics and Neural Field Coupled
LiDAR-Inertial Odometry</h1>

## Installation

### Platform requirement

* Ubuntu OS (tested on 20.04)

* With GPU (recommended)

* GPU memory requirement (> 8 GB recommended)


### 1. Set up conda environment

```
conda create --name knlio python=3.8
conda activate knlio
```

### 2. Install the key requirement PyTorch

```
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia 
```

The commands depend on your CUDA version (check it by `nvcc --version`). You may check the instructions [here](https://pytorch.org/get-started/previous-versions/).

### 3. Install other dependency

```
pip3 install -r requirements.txt
```

----

## Run KN-LIO

### Clone the repository

```
git clone https://gitee.com/alilili_ali/kn-lio
cd kn-lio
```


And then run:

```
python3 kn-lio.py ./config/viral/eee_01.yaml
```
