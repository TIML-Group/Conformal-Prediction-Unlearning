# Tackling Fake Forgetting through Uncertainty Quantification

[![preprint](https://img.shields.io/badge/arXiv-2410.07163-B31B1B)](https://arxiv.org/abs/2501.19403) 
[![ICML 2026](https://img.shields.io/badge/ICML-2026-blue)](https://openreview.net/group?id=ICML.cc/2026)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the official repo for the paper: [Tackling Fake Forgetting through Uncertainty Quantification](https://arxiv.org/abs/2501.19403).


##  News 
- [x] [2026.04] 👏👏 Accepted by [**ICML 2026**](https://openreview.net/forum?id=SaUgHTz5Yo).
- [x] [2025.06] 👏👏 Accepted by [**MuGen @ ICML 2025**](https://openreview.net/forum?id=wuGgok1Zyd).
- [x] [2025.01] 🚀🚀 Release the [**paper**](https://arxiv.org/abs/2501.19403).


## Abstract
Machine unlearning seeks to remove the influence of specified data from a trained model. While the unlearning accuracy provides a widely used metric for assessing unlearning performance, it falls short in assessing the reliability of forgetting. In this paper, we find that the forgetting data points misclassified by unlearning accuracy still have their ground truth labels included in the conformal prediction set from the uncertainty quantification perspective, leading to a phenomenon we term fake forgetting. To address this issue, we propose a novel metric CR, inspired by conformal prediction, that offers a more reliable assessment of forgetting quality. Building on these insights, we further propose an unlearning framework CPU that incorporates conformal prediction into the Carlini & Wagner adversarial attack loss, enabling the ground truth label to be effectively removed from the conformal prediction set. Through extensive experiments on image classification tasks, we demonstrate both the effectiveness of our proposed metric and the superior forgetting quality achieved by our framework.



## File Tree

Project file structure and description:

```
Conformal-Prediction-Unlearning
├─ README.md
├─ requirements.txt
├─ metrics	# package of our metrics (CR and MIACR)
│    ├─ CR.py
│    ├─ MIACR.py
├─ models	# package of models (ResNet-18 and Vit)
│    ├─ resnet.py
│    ├─ vit.py
├─ main_original_model.py
├─ main_unlearn.py
├─ main_unlearn_cpu.py
├─ main_evaluate.py
├─ unlearn.py
├─ unlearn_cpu.py
└─ utils.py
```

## Setup

Installation requirements are described in `requirements.txt`.

- Use pip:

  ```
  pip install -r requirements.txt
  ```

- Use anaconda:

  ```
  conda install --file requirements.txt
  ```

## Getting Started

Get an original model with the ResNet-18 or ViT architecture:

```
python main_original_model.py --model_name resnet18 --data_name cifar10 --data_dir ./data --batch_size 64 --num_epochs 200 --learning_rate 0.1 --num_classes 10 
```

To use the implemented logging, you’ll need a `wandb.ai` account. Alternatively, you can replace it with any logger of your preference.

To get an unlearning model with one of the existing unlearning methods, use the following command:

```
python main_unlearn.py --unlearn_name retrain --unlearn_type random --model_name resnet18 --data_name cifar10 --data_dir ./data --model_dir original_model.pth --num_epochs 200 --num_classes 10 --retain_ratio 0.9 --learning_rate 0.01
```

To get an unlearning model with our unlearning framework **CPU**, use the following command:

```
python main_unlearn_cpu.py --unlearn_name finetune --unlearn_type random --model_name resnet18 --data_name cifar10 --data_dir ./data --model_dir original_model.pth --num_epochs 20 --num_classes 10 --retain_ratio 0.9 --learning_rate 0.1 --delta 0.01 --alpha 0.05 --lamda 0.5
```

After unlearning the forget data, use `main_evaluate.py` to measure the unlearning model's performance by **CR** and **MIACR** metrics:

```
python main_evaluate.py --unlearn_name retrain --unlearn_type random --model_name resnet18 --data_name cifar10 --data_dir ./data --model_dir unlearning_model.pth --num_classes 10 --retain_ratio 0.9 --alphas 0.05
```

## How to Cite

```
@article{shi2025redefining,
  title={Redefining machine unlearning: A conformal prediction-motivated approach},
  author={Shi, Yingdan and Liu, Sijia and Wang, Ren},
  journal={arXiv preprint arXiv:2501.19403},
  year={2025}
}
```






















