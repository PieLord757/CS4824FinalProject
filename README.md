<div id="top">

<!-- HEADER STYLE: CLASSIC -->
<div align="center">

<img src="assets/drone_landing_logo.png" width="30%" style="position: relative; top: 0; right: 0;" alt="Project Logo"/>

# DRONE LANDING ZONE DETECTION – CS 4824 Final Project

<em>See Safe Ground Before You Touch Down</em>

<!-- BADGES -->
<img src="https://img.shields.io/github/license/PieLord757/CS4824FinalProject?style=flat&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
<img src="https://img.shields.io/github/last-commit/PieLord757/CS4824FinalProject?style=flat&logo=git&logoColor=white&color=0080ff" alt="last-commit">
<img src="https://img.shields.io/github/languages/top/PieLord757/CS4824FinalProject?style=flat&color=0080ff" alt="repo-top-language">
<img src="https://img.shields.io/github/languages/count/PieLord757/CS4824FinalProject?style=flat&color=0080ff" alt="repo-language-count">

<em>Built with the tools and technologies:</em><br>

<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/Jupyter-F37626.svg?style=flat&logo=Jupyter&logoColor=white" alt="Jupyter">
<img src="https://img.shields.io/badge/Roboflow-111827.svg?style=flat&logo=roboflow&logoColor=white" alt="Roboflow">
<img src="https://img.shields.io/badge/COCO%20Format-000000.svg?style=flat&logo=json&logoColor=white" alt="COCO">
<img src="https://img.shields.io/badge/PyTorch-EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white" alt="PyTorch"><br>
<img src="https://img.shields.io/badge/Blender-F5792A.svg?style=flat&logo=Blender&logoColor=white" alt="Blender">
<img src="https://img.shields.io/badge/Google%20Colab-F9AB00.svg?style=flat&logo=googlecolab&logoColor=white" alt="Google Colab">
<img src="https://img.shields.io/badge/OpenCV-27338e.svg?style=flat&logo=OpenCV&logoColor=white" alt="OpenCV">

</div>
<br>

---

## 📄 Table of Contents

- [Overview](#-overview)
- [Problem \& Motivation](#-problem--motivation)
- [What This Project Does](#-what-this-project-does)
- [Dataset \& Labeling](#-dataset--labeling)
- [Model \& Approach](#-model--approach)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#-prerequisites)
  - [Installation](#-installation)
  - [Data Preparation](#-data-preparation)
  - [Training](#-training)
  - [Inference](#-inference)
- [Results \& Evaluation](#-results--evaluation)
- [Challenges](#-challenges)
- [What We Learned](#-what-we-learned)
- [Limitations \& Future Work](#-limitations--future-work)
- [Tech Stack](#-tech-stack)
- [Contributors](#-contributors)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## ✨ Overview

This repository contains our **CS 4824 – Machine Learning** final project: a **computer vision system** that detects **safe landing zones for drones / VTOL craft** from images.

The goal is to build an end-to-end pipeline that:

- Takes **aerial or near-ground images** from drone-like viewpoints  
- Uses a modern **object detection model** to localize **landing zones**  
- Trains and evaluates the model using a **COCO-formatted dataset**  
- Can be extended to work with **synthetic scenes (Blender) and real-world photos**

We use **Roboflow’s RF-DETR** object detector and a **COCO dataset** to explore how well a model can learn to recognize “safe places to land” under diverse environments.

---

## 💡 Problem & Motivation

Autonomous drones and VTOL vehicles must often land in **unstructured, cluttered environments**:

- Rooftops, parking lots, courtyards, stadiums  
- Urban areas with people, cars, obstacles, wires  
- Remote areas with uneven ground or vegetation  

In many practical scenarios (delivery, inspection, search and rescue), the drone has to **choose a safe landing zone on its own**. Misclassification is dangerous:

- A **false positive** (unsafe area labeled safe) can lead to damage or injury  
- A **false negative** limits where the drone can land and reduces mission flexibility  

We ask:

> **Can we train a robust detector that reliably finds safe landing zones across diverse scenes and viewpoints?**

---

## 🏗️ What This Project Does

At a high level, this repository provides:

- 🧹 **Data Processing**  
  - Scripts to preprocess raw landing zone images  
  - Conversion of custom annotations into **COCO** format

- 🏷️ **Dataset Conversion**  
  - A script to **convert landing zone annotations → COCO JSON** compatible with RF-DETR and other detectors

- 🧠 **Model Training**  
  - A training script that uses **`config_landing_zone.yaml`** to train a model on the landing zone dataset  
  - A **Colab notebook** to run training in the cloud (GPU)

- ✅ **Verification & Setup**  
  - A helper script to verify that your environment, dependencies, and dataset paths are correctly configured

Overall, the repository is focused on **reproducible training** and **clean data handling** for the drone landing zone detection task.

---

## 🗂️ Dataset & Labeling

The dataset is stored as a compressed archive:

- **`coco-dataset.zip`** – COCO-style dataset containing:
  - Images of potential landing zones from diverse environments  
  - Bounding box annotations marking **landing zone regions**

### Sources & scenes

The project is designed to support images from:

- **Synthetic scenes** created in **Blender** (e.g., rooftops, fields, rooftops with helipads)  
- **Prompt-generated images** (e.g., Midjourney / similar tools)  
- **Real-world-like drone views** (city plazas, rooftops, open fields, campuses)

### Annotation format

The dataset uses **COCO** object detection format, with:

- `images` – metadata and file paths  
- `annotations` – bounding boxes + category id for **landing zones**  
- `categories` – class definitions (`landing_zone`, and optionally others)

You can regenerate or modify the COCO annotations using:

- **`convert_to_coco.py`** – converts raw annotation formats into a unified COCO JSON  
- **`process_landing_zones.py`** – optional additional cleaning / preparation logic

---

## 🧠 Model & Approach

This project is designed around **RF-DETR**, Roboflow’s implementation of the **DETR-style transformer-based object detector**:

- End-to-end detection (no anchors, no NMS)  
- Strong performance on crowded scenes  
- Works well with **COCO** and custom datasets  

The core configuration lives in:

- **`config_landing_zone.yaml`**

That config typically includes (you can open it and adjust):

- Paths to training / validation COCO files  
- Image sizes and augmentations  
- Model backbone / architecture choices  
- Training hyperparameters (epochs, batch size, learning rate, etc.)

Training logic is handled by:

- **`train_landing_zone.py`** – Python script for local training  
- **`train_landing_zone_colab.ipynb`** – Jupyter notebook for **Google Colab** training

---

## 📁 Project Structure

```text
CS4824FinalProject/
├── .gitattributes
├── .gitignore
├── coco-dataset.zip              # COCO dataset archive (images + JSON)
├── config_landing_zone.yaml      # RF-DETR / training configuration
├── convert_to_coco.py            # Convert raw annotations into COCO format
├── process_landing_zones.py      # Preprocess / clean landing zone datasets
├── train_landing_zone.py         # Main training script (Python)
├── train_landing_zone_colab.ipynb# Colab notebook for training in the cloud
└── verify_and_setup.py           # Environment + dataset verification helper
```
