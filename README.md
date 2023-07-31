# FastAno: Fast Fast Anomaly Detection via Spatio-temporal Patch Transformation (2022 WACV)

Authors: [Chaewon Park](https://github.com/codnjsqkr), Myeong Ah Cho, [Minhyeok Lee](https://github.com/Hydragon516), Sangyoun Lee

This repository contains the official implementation of Fastano (2022 WACV) [[Paper link]](https://openaccess.thecvf.com/content/WACV2022/html/Park_FastAno_Fast_Anomaly_Detection_via_Spatio-Temporal_Patch_Transformation_WACV_2022_paper.html)

![architecture](https://github.com/codnjsqkr/FastAno_official/assets/60251992/6a2e5570-2a3b-4c6c-8777-5d0113f4c8ba)

 We propose a new method called Fast Adaptive Patch Memory (FAPM) for real-time industrial anomaly detection. FAPM utilizes patch-wise and layer-wise memory banks that store the embedding features of images at the patch and layer level, respectively, which eliminates unnecessary repetitive computations. We also propose patch-wise adaptive coreset sampling for faster and more accurate detection. 

## Development setup

conda environment
```sh
git clone https://github.com/codnjsqkr/FastAno_official.git
conda create -n FAPM
conda activate FAPM

pip install -r requirements.txt

```
## Dataset
Please download UCSD Ped2 dataset from this [website](https://www.mvtec.com/company/research/datasets/mvtec-ad).

## Usage

Inference Code
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
