Privacy-Preserving Localization and Social Distance Monitoring with Low-Resolution Thermal Imaging
This repository contains the implementation for the paper:
"Privacy-Preserving Localization and Social Distance Monitoring with Low-Resolution Thermal Imaging and Deep Learning"
by Andrei Perov and Jens Heger
Available at: ScienceDirect

Overview
This study introduces a novel approach to leveraging low-power, low-resolution infrared (IR) sensors for detailed multi-person tracking in manufacturing environments. We curated a dataset capturing a wide range of human interactions, annotated for two key tasks:

Multiple-person localization

Social distance violation detection

Our method combines convolutional and recurrent neural networks to process spatiotemporal data. Notably:

We achieve 97.5% image-level accuracy in human localization using a novel segmentation-based approach.

We reach 91% macro-averaged accuracy in a four-class social distance violation classification task, emphasizing the importance of interpolation and kernel selection in model design.

Dataset
The Low-Resolution Infrared Sensor Dataset and supporting pipeline are available on Kaggle:
🔗 https://www.kaggle.com/datasets/andreyperov/lowresir-detect-and-distance

A sample of this dataset is included in the repository under the data_IR_final folder.

Repository Structure
Python_Files/
Full_pipeline_demonstration.ipynb
Full training and evaluation pipeline for both multi-person localization (spatiotemporal task) and social distance estimation.

GRAD_CAM.ipynb
Visualization of the social distance model using Grad-CAM for CNN explainability.

architectures.py
Contains all deep learning model architectures for both tasks.

baseline_algorithm.py
A non-deep learning baseline approach for comparison.

data_formatting.py
Functions for preprocessing and structuring the data for model input.

training.py
Utilities for training the deep learning models.

evaluation.py
Evaluation metrics and functions for model performance analysis.

grad_cam.py
Implementation of Grad-CAM visualization.

main.py
Main execution script for running the complete pipeline.

config.json
Configuration file for main.py execution settings.
