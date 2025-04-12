# Privacy-Preserving Localization and Social Distance Monitoring with Low-Resolution Thermal Imaging

This repository contains the implementation for the paper:  
**"Privacy-Preserving Localization and Social Distance Monitoring with Low-Resolution Thermal Imaging and Deep Learning"**  
by *Andrei Perov* and *Jens Heger*  
📄 [Read the paper on ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2212827124012575)

---

## 🧠 Overview

This study introduces a novel approach to leveraging low-power, low-resolution infrared (IR) sensors for detailed multi-person tracking in manufacturing environments. We curated a dataset capturing a wide range of human interactions, annotated for two key tasks:

- **Multiple-person localization**
- **Social distance violation detection**

Our method combines **convolutional** and **recurrent neural networks** to process spatiotemporal data. Key results:

- ✅ **97.5% image-level accuracy** in human localization using a novel segmentation-based approach  
- 📏 **91% macro-averaged accuracy** in a 4-class social distance classification task  

This highlights the importance of **interpolation methods** and **convolutional kernel selection** for effective social distance modeling.

---

## 📂 Dataset

The **Low-Resolution Infrared Sensor Dataset** and supporting pipeline are publicly available on Kaggle:

👉 [LowResIR Dataset on Kaggle](https://www.kaggle.com/datasets/andreyperov/lowresir-detect-and-distance)

A small sample of the dataset is included in this repository under the `data_IR_final/` directory.

---

## 📁 Repository Structure

### `Python_Files/`

| File | Description |
|------|-------------|
| `Full_pipeline_demonstration.ipynb` | Full training and evaluation pipeline for both multi-person localization (spatiotemporal task) and social distance estimation |
| `GRAD_CAM.ipynb` | Visualization of the social distance model using Grad-CAM (CNN explainability) |
| `architectures.py` | Deep learning architectures for localization and social distance models |
| `baseline_algorithm.py` | Non-deep learning baseline approach |
| `data_formatting.py` | Data preprocessing and formatting utilities |
| `training.py` | Training loop and optimization utilities |
| `evaluation.py` | Model evaluation functions and metrics |
| `grad_cam.py` | Grad-CAM visualization implementation |
| `main.py` | Main script to run the full pipeline |
| `config.json` | Configuration file for pipeline execution |

---

## 📌 Notes

- For full dataset access, please download from the [Kaggle link](https://www.kaggle.com/datasets/andreyperov/lowresir-detect-and-distance).
- The models and scripts are designed for reproducibility and extension for further research.

---

## 📬 Contact

For questions or collaborations, feel free to open an issue or contact the authors via the paper link above.
