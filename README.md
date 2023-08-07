# FastAno: Fast Anomaly Detection via Spatio-temporal Patch Transformation (2022 WACV)

Authors: [Chaewon Park](https://github.com/codnjsqkr), MyeongAh Cho, [Minhyeok Lee](https://github.com/Hydragon516), Sangyoun Lee

This repository contains the official implementation of FastAno (2022 WACV) [[Paper link]](https://openaccess.thecvf.com/content/WACV2022/html/Park_FastAno_Fast_Anomaly_Detection_via_Spatio-Temporal_Patch_Transformation_WACV_2022_paper.html)  
The inference code and our pre-trained weights are available. 

![architecture](https://github.com/codnjsqkr/FastAno_official/assets/60251992/f77d9fe0-3692-4160-802b-3c472d0d4336)
We propose spatial rotation transformation (SRT) and temporal mixing transformation (TMT) to generate irregular patch cuboids within normal frame cuboids in order to enhance the learning of normal features. Additionally, the proposed patch transformation is used only during the training phase, allowing our model to detect abnormal frames at fast speed during inference. Our model is evaluated on three anomaly detection benchmarks, achieving competitive accuracy and surpassing all the previous works in terms of speed.

## Model speed
<p align="center">
<img src="https://github.com/codnjsqkr/FastAno_official/assets/60251992/a6b9b02c-b4b7-4aee-a101-75b5b846c382.png" width="750" height="500"/>
</p>
Our inference speed is obtained by running FP32 model using a single Nvidia GeForce RTX 3090 without GPU warm-up. The FPS was computed by measuring the time from data preprocessing to calculating AUROC after all data had passed through the model, and then dividing it by the total number of frames in the test data folder. This process is found in the code.

## Development setup

- conda environment
```sh
git clone https://github.com/codnjsqkr/FastAno_official.git
conda create -n fastano python=3.7.9
conda activate fastano
pip install -r requirements.txt

```
## Dataset preparation
1. Create [data] folder inside FastAno_official 
2. Download UCSD Ped2 dataset from this [website](http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm) and place it in [data] folder.
3. The folder structure should look like this:  
   | Fastano_official  
   |---data  
   |-------ped2  
   |----------testing  
   |----------training  
   |----------ped2.mat  
   |---weights

## Inference with pre-trained weights

- Pre-trained weights trained on UCSD Ped2 is located in FastAno_official/weights/ped2_best.pth  
- Inference code
```sh
python main.py
```

## Citation
Cite below if you find this repository helpful to your project:
```sh
@InProceedings{Park_2022_WACV,
    author    = {Park, Chaewon and Cho, MyeongAh and Lee, Minhyeok and Lee, Sangyoun},
    title     = {FastAno: Fast Anomaly Detection via Spatio-Temporal Patch Transformation},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2022},
    pages     = {2249-2259}
}
```
