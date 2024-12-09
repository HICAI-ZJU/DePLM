# DePLM: Denoising Protein Language Models for Property Optimization

This repository is the official model and benchmark proposed in a paper: [InstructProtein: Aligning Human and Protein Language via Knowledge Instruction](https://neurips.cc/virtual/2024/poster/95517).

## Description

The central concept of DePLM revolves around perceiving the EI captured by PLMs as a blend of property-relevant and irrelevant information, with the latter akin to “noise” for the targeted property, necessitating its elimination. To achieve this, drawing inspiration from denoising diffusion models that refine noisy inputs to generate desired outputs, we devise a rank-based forward process to extend the diffusion model for denoising EI.

## Installation

```
>> git clone https://github.com/HICAI-ZJU/DePLM
>> cd DePLM
>> conda env create --file environment.yml
```

## Quick Start

We can train and test DePLM as follows.

```
>> bash ./scripts/schedule.sh
```

Here we use a deep mutational scanning (DMS) dataset - TAT_HV1BR_Fernandes_2016 - as an example. The program will run a training process with the default parameters.

To train on your own dataset, you need to provide DMS and structure data, place them in `./data`, and modify the data configuration file `./configs/data`.

## Citation

Please consider citing our paper if you find the code useful for your project.

```
@inproceedings{
    wang2024deplm,
    title={De{PLM}: Denoising Protein Language Models for Property Optimization},
    author={Zeyuan Wang and Keyan Ding and Ming Qin and Xiaotong Li and Xiang Zhuang and Yu Zhao and Jianhua Yao and Qiang Zhang and Huajun Chen},
    booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
    year={2024},
    url={https://openreview.net/forum?id=MU27zjHBcW}
}
```