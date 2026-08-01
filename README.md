# IMUZero: Zero-Shot Human Activity Recognition by Language-Based Cross Modality Fusion

<p align="center">
  <img src="assets/teaser.png" width="900">
</p>

<p align="center">
  <a href="https://dl.acm.org/doi/10.1145/3770669">
    <img src="https://img.shields.io/badge/Paper-ACM%20IMWUT-red">
  </a>
  <a href="https://github.com/Was-Lab/IMUZero">
    <img src="https://img.shields.io/badge/Code-Coming%20Soon-black">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-green.svg">
  </a>
</p>

This repository contains the official implementation of:

**IMUZero: Zero-Shot Human Activity Recognition by Language-Based Cross Modality Fusion**

Published in:

**Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT), 2025**

[Paper](https://dl.acm.org/doi/10.1145/3770669)

---

## Abstract

Human Activity Recognition (HAR) using wearable inertial sensors has achieved significant progress with deep learning techniques. However, most existing approaches require large-scale labeled data and struggle to recognize activities that are not observed during training.

In this work, we introduce **IMUZero**, a zero-shot human activity recognition framework that enables recognition of unseen activities by leveraging semantic knowledge from natural language.

IMUZero bridges the modality gap between inertial sensor signals and activity semantics through language-based cross-modal fusion. Specifically, we employ large language models to generate fine-grained semantic attributes describing human activities and align these semantic representations with IMU features.

Furthermore, we introduce a sensor-aware representation learning strategy based on channel shuffle order constraints to address the inherent axial bias of inertial sensors.

Extensive experiments demonstrate that IMUZero achieves strong generalization capability on zero-shot HAR benchmarks.

---

## Method Overview

<p align="center">
  <img src="assets/framework.png" width="900">
</p>

The framework consists of three major components:

### 1. IMU Representation Encoder

Given raw inertial measurements, the encoder learns discriminative motion representations from sensor signals.

### 2. Language-based Semantic Representation

Large language models are leveraged to generate fine-grained descriptions and semantic attributes for human activities.

These descriptions provide transferable knowledge beyond predefined activity labels.

### 3. Cross-modal Feature Alignment

The learned IMU representations are aligned with language semantic embeddings, enabling knowledge transfer from seen activities to unseen activities.

---

# Contributions

The main contributions of this work are:

* We propose **IMUZero**, the first language-based zero-shot framework for human activity recognition from IMU sensors.
* We introduce an LLM-assisted semantic attribute generation strategy for fine-grained activity representation.
* We design a cross-modal fusion mechanism to bridge inertial sensing and natural language modalities.
* We propose a sensor-aware learning constraint to alleviate IMU axial bias.
* We demonstrate superior generalization performance across multiple HAR benchmarks.

---

# Installation

## Environment

The implementation is tested with:

```
Python >= 3.8
PyTorch >= 1.12
CUDA >= 11.0
```

Install dependencies:

```bash
git clone https://github.com/Was-Lab/IMUZero.git

cd IMUZero

pip install -r requirements.txt
```

---

# Dataset Preparation

We evaluate IMUZero on public human activity recognition datasets.

Please download the datasets from their official sources and organize them as follows:

```
data/
├── dataset_name_1/
├── dataset_name_2/
├── dataset_name_3/
└── dataset_name_4/
```

Update the dataset paths in:

```
configs/
```

---

# Training

To train IMUZero:

```bash
python train.py \
    --config configs/<dataset>.yaml
```

Example:

```bash
python train.py \
    --config configs/pamap2.yaml
```

---

# Evaluation

Evaluate a trained checkpoint:

```bash
python test.py \
    --checkpoint path/to/checkpoint
```

---

# Main Results

Zero-shot human activity recognition performance:

|      Method      | Dataset A | Dataset B | Dataset C | Dataset D |
| :--------------: | :-------: | :-------: | :-------: | :-------: |
| Previous Methods |     -     |     -     |     -     |     -     |
|    **IMUZero**   |   **-**   |   **-**   |   **-**   |   **-**   |

More details are available in the paper.

---

# Ablation Studies

We investigate the contribution of:

* Language-based semantic attributes
* Cross-modal alignment
* Sensor-aware constraints

Detailed ablation results are provided in the paper.

---

# Citation

If you find IMUZero useful for your research, please cite:

```bibtex
@article{su2025imuzero,
  title={IMUZero: Zero-Shot Human Activity Recognition by Language-Based Cross Modality Fusion},
  author={Su, Jie and Ge, Fengtong and Wen, Zhenyu and Li, Taotao and Bai, Yang and Zhou, Yejian and Zhang, Xiaoqin},
  journal={Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  volume={9},
  number={4},
  pages={1--28},
  year={2025},
  publisher={ACM},
  doi={10.1145/3770669}
}
```

---

# Acknowledgements

We thank the research community for their contributions to:

* Human Activity Recognition
* Wearable Sensing
* Zero-shot Learning
* Multimodal Representation Learning

---

# License

This project is released for academic research purposes.

---

# Contact

For questions and discussions, please open an issue or contact the authors.

