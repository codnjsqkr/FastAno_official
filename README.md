# FastAno: Fast Fast Anomaly Detection via Spatio-temporal Patch Transformation (2022 WACV)

Authors: [Chaewon Park](https://github.com/codnjsqkr), Myeong Ah Cho, [Minhyeok Lee](https://github.com/Hydragon516), Sangyoun Lee

This repository contains the official implementation of FastAno (2022 WACV) [[Paper link]](https://openaccess.thecvf.com/content/WACV2022/html/Park_FastAno_Fast_Anomaly_Detection_via_Spatio-Temporal_Patch_Transformation_WACV_2022_paper.html)  
The inference code and our pre-trained weights are available. 

![architecture](https://github.com/codnjsqkr/FastAno_official/assets/60251992/6a2e5570-2a3b-4c6c-8777-5d0113f4c8ba)  
We propose spatial rotation transformation (SRT) and temporal mixing transformation (TMT) to generate irregular patch cuboids within normal frame cuboids in order to enhance the learning of normal features. Additionally, the proposed patch transformation is used only during the training phase, allowing our model to detect abnormal frames at fast speed during inference. Our model is evaluated on three anomaly detection benchmarks, achieving competitive accuracy and surpassing all the previous works in terms of speed.

<center><img src="https://github.com/codnjsqkr/FastAno_official/assets/60251992/9a798b61-15d5-42e6-bcb8-81efab9f97b6.png" width="600" height="450"/></center>

## Development setup

- conda environment
```sh
git clone https://github.com/codnjsqkr/FastAno_official.git
conda create -n fastano python=3.7.9
conda activate fastano
pip install -r requirements.txt

```
## Dataset
1. Create [data] folder inside FastAno_official
2. Download UCSD Ped2 dataset from this [website](http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm) and place it in [data] folder.
3. It should look like this:  
   | Fastano_official  
   |---data  
   |-------ped2  
   |----------testing  
   |----------training  
   |----------ped2.mat

## Inference with pre-trained weights

- Pre-trained weights trained on UCSD Ped2 is located in Fastano/weights/ped2_best.pth  
- Inference code
```sh
python main.py
```

## Citation
Cite as below if you find this repository is helpful to your project:
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
