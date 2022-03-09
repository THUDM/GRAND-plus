# GRAND+

This is a PyTorch implementation of GRAND+ for scalable graph-based semi-supervised learning:

[GRAND+: Scalable Graph Random Neural Networks]


## Datasets
This repo contains `Cora`, `Citeseer` and `Pubmed` datasets under the path `dataset/citation/`. The other datasets used in the paper (including `AMiner-CS`, `Reddit`, `Amazon2M` and `MAG-Scholar-C`) can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1LV8kMRnQENQnwi6qtbycTgVAEGX8rxQv?usp=sharing) or [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/d8194be5640242759671/). To run model on these datasets, you should download the corresponding zip file, uncompress it and put it under `dataset/`. 

You can directly download the zip file of each dataset with the following scripts:

- Download datasets from Google Drive
```
pip install gdown
gdown --id 1G9Wn1OaqMYpkNmbOESYUFrDgzo0Be0-L -O dataset/aminer.zip
gdown --id 1KauMd-AJXyD6KQQnf4vySjRZEOgWQYvx -O dataset/reddit.zip
gdown --id 1uItY1AGywFv4nSSFpqBaTEUoDn3w414B -O dataset/Amazon2M.zip
gdown --id 1VKHFQfRXkkVShE6d4hA9dImXZalz49qa -O dataset/mag_scholar_c.npz
```

- Download datasets from Tsinghua Cloud 
```
python scripts/download.py --url https://cloud.tsinghua.edu.cn/f/629a605e453b40fc9a93/?dl=1 --path dataset --fname aminer.zip
python scripts/download.py --url https://cloud.tsinghua.edu.cn/f/384be92876ed4127aa3c/?dl=1 --path dataset --fname reddit.zip
python scripts/download.py --url https://cloud.tsinghua.edu.cn/f/7c867cef16214fe1a30b/?dl=1 --path dataset --fname Amazon2M.zip
python scripts/download.py --url https://cloud.tsinghua.edu.cn/f/5e5c9d8833a143d5abb4/?dl=1 --path dataset --fname mag_scholar_c.npz
```

## Requirements
- g++ 7.5.0
- [pybind11](https://pybind11.readthedocs.io/en/stable/installing.html)
- networkx 2.5
- numpy 1.19.2
- scikit_learn 1.0.2
- scipy 1.5.2
- torch 1.8.1 (cuda 10.2)
- [torch_scatter 2.0.6](https://github.com/rusty1s/pytorch_scatter)

## Compilation
`make clean && make`

## Running the code
- Reproduce the results on Cora: `sh scripts/run_cora.sh` 
- Reproduce the results on Citeseer: `sh scripts/run_citeseer.sh` 
- Reproduce the results on Pubmed: `sh scripts/run_pubmed.sh` 
- Reproduce the results on AMiner-CS: `sh scripts/run_aminer.sh` 
- Reproduce the results on Reddit: `sh scripts/run_reddit.sh` 
- Reproduce the results on Amazon2M: `sh scripts/run_amazon2m.sh` 
- Reproduce the results on MAG-Scholar-C: `sh scripts/run_mag.sh` 

## Cite

If you find this work is helpful to your research, please consider citing our paper:

```
@inproceedings{feng2022grand+,
  title={GRAND+: Scalable Graph Random Neural Networks},
  author={Feng, Wenzheng and Dong, Yuxiao and Huang, Tinglin and Yin, Ziqi and Cheng, Xu and Kharlamov, Evgeny and Tang, Jie},
  booktitle={Proceedings of the ACM Web Conference 2022 (WWW’22)},
  year={2022}
}
```
