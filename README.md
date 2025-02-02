Deep Learning-Based Histological Image Analysis for Cancer Detection

Overview

Early and accurate cancer detection is a critical challenge in modern medicine, with significant implications for patient survival rates. This project focuses on the design and implementation of deep learning and computer vision solutions to address this challenge by analyzing histological images.

Project Scope

We developed and evaluated customized models and pre-trained convolutional neural networks (CNNs), including DenseNet121, VGG16, and ResNet50, for the classification of multiple cancer types, such as lung, colon, oral, and gastrointestinal cancer.

Dataset

The datasets used in this research come from well-recognized sources, including the LC25000 dataset and other specialized histological image datasets. These datasets were preprocessed and augmented to enhance their representativeness and balance.

Model Optimization

Hyperparameter optimization was performed using Bayesian search, yielding outstanding accuracy:

DenseNet121 achieved 100% accuracy in lung and gastrointestinal cancer classification.

The customized model reached 99.9% accuracy for colon cancer classification, surpassing the general model with 98.46% accuracy.

Model Explainability

To ensure clinical applicability, Grad-CAM was employed for model interpretation, providing intuitive visualizations of the most relevant image regions. This enhances trust in the system and facilitates its integration into medical workflows.

Performance Comparison

Compared to prior studies, such as Merabet (2024) and Ahmed (2023), the models in this project demonstrated high competitiveness:

DenseNet121 outperformed other approaches in lung and gastrointestinal cancer classification.

The customized model exhibited advantages in specific contexts, particularly in colon cancer classification.

However, for oral cancer detection, results indicate the need for expanded datasets and hybrid approaches to enhance model robustness.

Conclusion

The results obtained are not only competitive with those reported in the literature but also highlight the potential of deep learning technologies to transform clinical practice. These findings underscore the feasibility of integrating AI-based diagnostic tools to improve the accuracy and efficiency of cancer detection, paving the way for advancements in modern medical applications.
