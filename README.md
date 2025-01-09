# 🚗 Deep Learning - Car Damage Detection with CNNs
This repository contains an end-to-end pipeline for detecting and classifying car damages using Convolutional Neural Networks (CNNs) and pretrained models integrated with APIs and UI for deployment.
## Overview
This project aims to classify car damages into multiple categories like front crushed, rear breakages, normal conditions, etc., leveraging state-of-the-art deep learning technologies.
### Features:
- **Pretrained Models**: Transfer learning with **ResNet50**, **EfficientNet** with fine-tuning for task-specific accuracy.
- **Custom CNN**: Built from scratch and compared with pretrained methods.
- **Hyperparameter Tuning**: Automated using **Optuna**!
- **FastAPI**: An API endpoint for real-time predictions.
- **Streamlit Web App**: Interactive UI for demo purposes.

### Technical Highlights:
- **Data Augmentation**: Extensive transformations for robust model generalization.
- **GPU Training**: Utilized PyTorch for CUDA-accelerated training.
- **Inference Pipeline**: Image preprocessing, prediction, and visualization!
- **Deployment**: FastAPI endpoint for integration with applications and services.

### Usage:
1. **Clone Repository**
2. **Train from Scratch or Use Pretrained Weights**
3. **Deploy Locally or on Cloud** 🌐
