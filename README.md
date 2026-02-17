# Salary Prediction from CV Data (Keras Multi-Input Model)

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/keras-tf.keras-red.svg)](https://keras.io/)

A deep learning project for salary regression based on structured and textual resume (CV) data.

The system combines:
- 📊 Structured tabular features (age, city, experience, employment type, schedule)
- 🎓 Education text (Bag-of-Words)
- 💼 Work experience text (Bag-of-Words)
- 🏷 Position title text (Bag-of-Words)

Two model variants are implemented:
- Full multi-branch model
- Simplified model (tabular + position only)

---

## 📁 Project Structure

```bash
├── constants.py        # Global configuration and hyperparameters
├── dataset.py          # Data download, preprocessing, feature engineering
├── models.py           # Multi-input neural network architectures
├── train.py            # Training scripts
└── README.md           # Project documentation
```

---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Model Architecture](#model-architecture)
---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Korny998/Analysis-of-candidates-resumes.git
cd Analysis-of-candidates-resumes
```

2. Create a virtual environment:

```bash
python -m venv venv
```

3. Activate the environment:

Windows:

```bash
venv\Scripts\activate
```

Linux / macOS:

```bash
source venv/bin/activate
```

4. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

**To train the full and simplified models:**

```python
python train.py
```

The script will automatically:
1. Download the CV dataset (if not present).
2. Preprocess structured and textual features.
3. Build Bag-of-Words matrices.
4. Scale salary target values.
5. Train both models.
6. Generate salary predictions (with inverse scaling).

## Dataset

* The dataset is automatically downloaded from:

```python
https://storage.yandexcloud.net/academy.ai/cv_100000.csv
```

It contains approximately 100,000 CV records with structured and semi-structured fields.

## Feature Engineering

The preprocessing pipeline performs:

1. Structured Feature Encoding
- City → One-Hot Encoding
- Age → Bucketization + One-Hot
- Experience → Bucketization + One-Hot
- Employment type → One-Hot
- Work schedule → One-Hot

2. Text Processing

The following fields are extracted from JSON structures:
- Education history
- Work experience history
- Position name

Each text field is converted into a Bag-of-Words matrix using:

```python
Tokenizer(num_words=NUM_WORDS)
```

Default vocabulary size:

```bash
NUM_WORDS = 3000
```

## Model Architecture

*1. Full Multi-Input Model*

- Layers:
    - Branch 1 – Tabular Features: Input → Dense(20) → Dense(500) → Dense(200)
    - Branch 2 – Education Text: Input → Dense(20) → Dense(200) → Dropout(0.3)
    - Branch 3 – Work Experience Text: Input → Dense(20) → Dense(200) → Dropout(0.3)
    - Branch 4 – Position Title: Input → Dense(20) → Dense(200) → Dropout(0.3)
    - Fusion Layer: Concatenate → Dense(30) → Dropout(0.5) → Dense(1)
    - Output: 1 neuron -> 1 neuron -> Linear activation

*2. Simplified Model*

Gated Recurrent Units with stronger temporal modeling.

- Layers:
    - (Tabular branch + Position branch)
    - Concatenate
    - Dense(30)
    - Dropout(0.5)
    - Dense(1)
