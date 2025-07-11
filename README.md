# ISTDiffuser: Infrared Small Target Image Generation via Conditional Denoising Diffusion with Contrastive Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


---

## Introduction

ISTDiffuser is a conditional denoising diffusion framework with contrastive learning for infrared small target image generation. This work pioneers the application of diffusion models in infrared small target detection, achieving high-fidelity image generation through innovative conditional denoising mechanisms and contrastive learning strategies.

## 🚀 News

- ✅ **[2025.07]** Our paper has been officially accepted!  
- ✅ **[2025.07]** Source code and pre-trained models are now publicly available.  
- ⏳ **IST-SIRST dataset will be released soon.**

## 🔧 Installation
```bash
git clone https://github.com/Tianzishu/istdiffuser.git
cd istdiffuser
pip install -r requirements.txt

## Prerequisite
* Tested on Ubuntu 20.04, with Python 3.8, PyTorch 1.8.1, Torchvision 0.9.1, CUDA 11.7, and 1x NVIDIA A800. 

* **NUDT-SIRST** &nbsp; [[download]](https://github.com/YeRen123455/Infrared-Small-Target-Detection) &nbsp; [[paper]](https://ieeexplore.ieee.org/abstract/document/9864119)

* **IRSTD-1K** &nbsp; [[download]](https://github.com/RuiZhang97/ISNet) &nbsp; [[paper]](https://ieeexplore.ieee.org/document/9880295)
---

## 📦 Pretrained Models
You can download our pre-trained weights via [BaiduYun Drive](https://pan.baidu.com/s/1krWT2I4OPlC-_dlV8Z16PQ?pwd=unny).(Includes checkpoints for NUDT-SIRST) 

## Results Demonstration

*Note: Left: Original infrared images; Right: Enhanced small target generation results by ISTDiffuser*

### Performance on NUDT-SIRST Dataset
![NUDT-SIRST Comparison](nudt_comparison_collage.webp)

### Performance on IRSTD-1K Dataset
![IRSTD-1K Comparison](1k_comparison_collage.webp)

