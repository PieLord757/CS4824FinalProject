```{=html}
<div id="top">
```
```{=html}
<!-- HEADER STYLE: CLASSIC -->
```
::: {align="center"}
`<img src="assets/drone_landing_logo.png" width="30%" style="position: relative; top: 0; right: 0;" alt="Project Logo"/>`{=html}

# DRONE LANDING ZONE DETECTION -- CS 4824 Final Project

`<em>`{=html}See Safe Ground Before You Touch Down`</em>`{=html}

```{=html}
<!-- BADGES -->
```
`<img src="https://img.shields.io/github/license/PieLord757/CS4824FinalProject?style=flat&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">`{=html}
`<img src="https://img.shields.io/github/last-commit/PieLord757/CS4824FinalProject?style=flat&logo=git&logoColor=white&color=0080ff" alt="last-commit">`{=html}
`<img src="https://img.shields.io/github/languages/top/PieLord757/CS4824FinalProject?style=flat&color=0080ff" alt="repo-top-language">`{=html}
`<img src="https://img.shields.io/github/languages/count/PieLord757/CS4824FinalProject?style=flat&color=0080ff" alt="repo-language-count">`{=html}

`<em>`{=html}Built with the tools and
technologies:`</em>`{=html}`<br>`{=html}

`<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white" alt="Python">`{=html}
`<img src="https://img.shields.io/badge/Jupyter-F37626.svg?style=flat&logo=Jupyter&logoColor=white" alt="Jupyter">`{=html}
`<img src="https://img.shields.io/badge/Roboflow-111827.svg?style=flat&logo=roboflow&logoColor=white" alt="Roboflow">`{=html}
`<img src="https://img.shields.io/badge/COCO%20Format-000000.svg?style=flat&logo=json&logoColor=white" alt="COCO">`{=html}
`<img src="https://img.shields.io/badge/PyTorch-EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white" alt="PyTorch">`{=html}`<br>`{=html}
`<img src="https://img.shields.io/badge/Blender-F5792A.svg?style=flat&logo=Blender&logoColor=white" alt="Blender">`{=html}
`<img src="https://img.shields.io/badge/Google%20Colab-F9AB00.svg?style=flat&logo=googlecolab&logoColor=white" alt="Google Colab">`{=html}
`<img src="https://img.shields.io/badge/OpenCV-27338e.svg?style=flat&logo=OpenCV&logoColor=white" alt="OpenCV">`{=html}
:::

`<br>`{=html}

------------------------------------------------------------------------

## 📄 Table of Contents

-   [Overview](#-overview)\
-   [Problem & Motivation](#-problem--motivation)\
-   [What This Project Does](#-what-this-project-does)\
-   [Dataset & Labeling](#-dataset--labeling)\
-   [Model & Approach](#-model--approach)\
-   [Project Structure](#-project-structure)\
-   [Getting Started](#-getting-started)
    -   [Prerequisites](#-prerequisites)\
    -   [Installation](#-installation)\
    -   [Data Preparation](#-data-preparation)\
    -   [Training](#-training)\
    -   [Inference](#-inference)\
-   [Results & Evaluation](#-results--evaluation)\
-   [Challenges](#-challenges)\
-   [What We Learned](#-what-we-learned)\
-   [Limitations & Future Work](#-limitations--future-work)\
-   [Tech Stack](#-tech-stack)\
-   [Contributors](#-contributors)\
-   [License](#-license)\
-   [Acknowledgments](#-acknowledgments)

------------------------------------------------------------------------

## ✨ Overview

This repository contains our **CS 4824 -- Machine Learning Final
Project**:\
a **computer vision model** that detects **safe landing zones** for
drones and VTOL aircraft.

We use:

-   **RF-DETR** (Roboflow's DETR-based detector)
-   **COCO-format dataset**
-   **Synthetic scenes + real drone-perspective images**
-   **Python + Colab training pipeline**

The goal is simple:

> **Teach a model to identify where a drone can safely land, across many
> environments and angles.**

------------------------------------------------------------------------

## 💡 Problem & Motivation

Drones often need to land in **unstructured or unknown areas**:

-   Rooftops\
-   Concrete pads\
-   Parking lots\
-   Courtyards\
-   Remote fields

A wrong prediction can cause:

-   🚨 Damage to drone\
-   🚨 Safety hazards\
-   🚨 Failed missions

Landing zones are **not objects**---they are **regions**.\
This makes the problem more challenging than detecting cars, people, or
signs.

Our project explores:

-   How to define the concept of a "good landing zone"\
-   How to annotate these zones\
-   Whether a model can learn this reliably

------------------------------------------------------------------------

## 🏗️ What This Project Does

Our project implements a complete workflow:

### ✔ Data cleaning

### ✔ Convert landing zone annotations → COCO format

### ✔ RF-DETR training logic

### ✔ Training via Google Colab or local machine

### ✔ Clean project structure for reproduction

### ✔ Evaluation plan & visualizations

------------------------------------------------------------------------

## 🗂️ Dataset & Labeling

The dataset is included as:

    coco-dataset.zip

It contains:

-   Aerial + mixed perspective images
-   Synthetic scenes (Blender-generated)
-   AI-assisted synthetic scenes
-   Manually annotated landing zone bounding boxes

Labels follow **COCO detection format**:

    images/
    annotations/
    instances_train.json
    instances_val.json

### File utilities:

-   `convert_to_coco.py` --- Converts raw labels to COCO JSON\
-   `process_landing_zones.py` --- Preprocessing / normalization

------------------------------------------------------------------------

## 🧠 Model & Approach

We use **Roboflow's RF-DETR**, an improved DETR-based detector.

Why RF-DETR?

-   No anchors\
-   No NMS\
-   Global attention helps detect regions\
-   Good for messy real-world scenes\
-   Works perfectly with COCO format

Training is configured using:

    config_landing_zone.yaml

Training is executed via:

    train_landing_zone.py

Or using the Colab notebook:

    train_landing_zone_colab.ipynb

------------------------------------------------------------------------

## 📁 Project Structure

    CS4824FinalProject/
    ├── coco-dataset.zip
    ├── config_landing_zone.yaml
    ├── convert_to_coco.py
    ├── process_landing_zones.py
    ├── train_landing_zone.py
    ├── train_landing_zone_colab.ipynb
    ├── verify_and_setup.py
    └── README.md

------------------------------------------------------------------------

## 🚀 Getting Started

### Prerequisites

-   Python ≥ 3.9

### Installation

``` bash
git clone https://github.com/PieLord757/CS4824FinalProject
cd CS4824FinalProject
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 📦 Data Preparation

``` bash
unzip coco-dataset.zip -d data
```

------------------------------------------------------------------------

## 🏋️ Training

``` bash
python train_landing_zone.py --config config_landing_zone.yaml
```

------------------------------------------------------------------------

## 📊 Results & Evaluation

Evaluation includes:

-   mAP@0.5\
-   mAP@0.5:0.95

------------------------------------------------------------------------

## 👥 Contributors

-   **Stephen Nguyen**\
-   **Team Member 2**\
-   **Team Member 3**

------------------------------------------------------------------------

## 📜 License

MIT License

------------------------------------------------------------------------

## ✨ Acknowledgments

-   CS 4824 Course Staff\
-   Roboflow RF-DETR\
-   Blender Community\
-   Google Colab

`<br>`{=html}

::: {align="left"}
`<a href="#top">`{=html}⬆ Return to top`</a>`{=html}
:::
