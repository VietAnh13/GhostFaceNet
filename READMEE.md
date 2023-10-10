# GhostFaceNets

## Original README
[Original README](https://github.com/HamadYA/GhostFaceNets#readme)

## Hardware configuration
CPU: [Intel® Core™ i5-10300H](https://www.intel.vn/content/www/vn/vi/products/sku/201839/intel-core-i510300h-processor-8m-cache-up-to-4-50-ghz/specifications.html)
GPU: [GeForce GTX 1650 Ti](https://www.nvidia.com/vi-vn/geforce/gaming-laptops/compare-16-series/)

## Note
Install [VSCode](https://code.visualstudio.com/)
Install [Anaconda](https://www.anaconda.com/download)

Update [NVDIA Driver](https://www.nvidia.com/download/index.aspx)
Install [Cuda Toolkit 11.2.0](https://developer.nvidia.com/cuda-toolkit-archive)
Download [cuDNN Library 11.2](https://developer.nvidia.com/cudnn)

You can refer to the detailed installation process [CUDA Install Guide](https://github.com/sithu31296/CUDA-Install-Guide#readme)

Please follow [Install TensorFlow with pip](https://www.tensorflow.org/install/pip#windows-native)

## Enviroment
Create a conda environment from the yml file and activate it as follows
```
conda env create -f environment.yml
conda activate ghostface-net
```

## Model
[GhostFaceNetV1-1.3-2 (A)](https://github.com/HamadYA/GhostFaceNets/releases/download/v1.5/GN_W1.3_S2_ArcFace_epoch48.h5)

## Test
**[zip_step.py](zip_step.py)** script
```
python zip_step.py -d ./path/to/data/folder -c ./path/to/csv/file -m ./path/to/model -B ./path/to/save/bin
```