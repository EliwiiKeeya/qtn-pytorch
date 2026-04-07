# QTN-Pytorch

本项目代码遵循 [GNU GPLv3 开源协议](./LICENSE)。

## 项目简介
本项目为 [QTN: Quaternion Transformer Network for Hyperspectral Image Classification](https://doi.org/10.1109/TCSVT.2023.3283289) 论文的 PyTorch 复现。原论文作者为 Yang, Xiaofei 等，发表于 IEEE Transactions on Circuits and Systems for Video Technology, 2023。

本仓库仅为论文方法的个人复现，非原作者官方实现。

## 环境依赖
- Python 3.12.13
- torch 2.11.0
- CUDA 13.2（如有NVIDIA GPU，建议使用对应CUDA版本）
- 其他依赖见 requirements.txt

> 注：本项目开发环境为 3060 Laptop + CUDA 13.2 + Python 3.12.13 + torch 2.11.0。深度学习框架的版本兼容性较严格，建议尽量使用 requirements.txt 中指定的版本。

## 依赖与致谢
- 本项目部分功能依赖 [Pytorch-Quaternion-Neural-Networks](https://github.com/Orkis-Research/Pytorch-Quaternion-Neural-Networks)（MIT License），感谢其开源贡献。


## 运行方法

```bash
# 安装依赖
pip install -r requirements.txt

# 运行主程序
python -u main.py
```

---

## 论文引用（BibTeX）
```bibtex
@ARTICLE{10144788,
	author={Yang, Xiaofei and Cao, Weijia and Lu, Yao and Zhou, Yicong},
	journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
	title={QTN: Quaternion Transformer Network for Hyperspectral Image Classification}, 
	year={2023},
	volume={33},
	number={12},
	pages={7370-7384},
	keywords={Transformers;Hyperspectral imaging;Image classification;Quaternions;Feature extraction;Convolutional neural networks;Hyperspectral image classification;convolution neural network;transformer network;quaternion transformer network (QTN)},
	doi={10.1109/TCSVT.2023.3283289}
}
```

---


如有问题欢迎提交 issue。
