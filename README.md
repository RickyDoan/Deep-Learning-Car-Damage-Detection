# 🚗 Deep Learning - Car Damage Detection with CNNs
This repository contains an end-to-end pipeline for detecting and classifying car damages using Convolutional Neural Networks (CNNs) and pretrained models integrated with APIs and UI for deployment.
![FF6CD62B-8F0C-4264-BD7F-C54ED7EEFD06_1_105_c](https://github.com/user-attachments/assets/2f535783-655b-4717-88d9-7561155aa68c)

## Overview
This project aims to classify car damages into multiple categories like front crushed, rear breakages, normal conditions, etc., leveraging state-of-the-art deep learning technologies.
### Features:
- **Pretrained Models**: Transfer learning with **ResNet50**, **EfficientNet** with fine-tuning for task-specific accuracy.
- **Custom CNN**: Built from scratch and compared with pretrained methods.
- **Hyperparameter Tuning**: Automated using **Optuna**!
- **FastAPI**: An API endpoint for real-time predictions.

### Technical Highlights:
- **Data Augmentation**: Extensive transformations for robust model generalization.
- **GPU Training**: Utilized PyTorch for CUDA-accelerated training on google collab
- **Inference Pipeline**: Image preprocessing, prediction, and visualization!

### Usage:
1. **Clone Repository**
2. **Train from Scratch or Use Pretrained Weights**

### **Evaluation**: 
* Confusion matrix and classification reports validate the models.
* Accuracy is around 80%
