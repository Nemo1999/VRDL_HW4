# VRDL_HW4
This is Homework4 Project for NYCU VRDL class
I trained MSRResNet and upscale testing images with scaling factor 3x

# Reference
- [Super Resolution with SRResnet](https://arxiv.org/abs/1609.04802)
- [KAIR Repo](https://github.com/cszn/KAIR)


# Prediction Results
PSNR : **28.1258**

<details>
<summary>Click to toggle Result Visualization</summary>

|LR (input) 	|  HR (generated)	|
|:----------:	|:--------:	|
|  <img width="500" src="https://github.com/Nemo1999/VRDL_HW4/blob/master/Prediction_Results/LR/00.png"/>  |  <img width="500" src="https://github.com/Nemo1999/VRDL_HW4/blob/master/Prediction_Results/HR/00_pred.png"/>      |
|  <img width="500" src="https://github.com/Nemo1999/VRDL_HW4/blob/master/Prediction_Results/LR/01.png"/>       	|  <img width="500" src="https://github.com/Nemo1999/VRDL_HW4/blob/master/Prediction_Results/HR/01_pred.png"/>              	|
|  <img width="500" src="https://github.com/Nemo1999/VRDL_HW4/blob/master/Prediction_Results/LR/02.png"/>       	|  <img width="500" src="https://github.com/Nemo1999/VRDL_HW4/blob/master/Prediction_Results/HR/02_pred.png"/>              	|
|  <img width="500" src="https://github.com/Nemo1999/VRDL_HW4/blob/master/Prediction_Results/LR/03.png"/>       	|  <img width="500" src="https://github.com/Nemo1999/VRDL_HW4/blob/master/Prediction_Results/HR/03_pred.png"/>              	|
|  <img width="500" src="https://github.com/Nemo1999/VRDL_HW4/blob/master/Prediction_Results/LR/04.png"/>       	|  <img width="500" src="https://github.com/Nemo1999/VRDL_HW4/blob/master/Prediction_Results/HR/04_pred.png"/>              	|

</details>

# Prerequist
To run the project
- Install [pytorch](https://pytorch.org/) version 10.2 (earlier versions may also work)
- Install dependencies
```bash
make install
```
# Reproduce 

- ## Download Dataset
```bash
make download_data
```
The data will be automatically downloaded and unzipped in   `VRDL_HW4/data`
- ## Reproduce HR images
```bash
make reproduce
```
The script will automatic download trained model checkpoints from google drive and run evaluation on testing dataset.

The resulting images will be stored at `VRDL_HW4/KAIR_Repo/superresolution/msrresnet_psnr/images/evaluation{iterations}/`

- ## Train from scratch
```bash
make train
```
The training result will be stored at `VRDL_HW4/KAIR_Repo/superresolution/`

