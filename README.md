# Fingerprint-Authentication-System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.7+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-Educational-blue?style=for-the-badge)

**A high-performance biometric system that enrolls users via fingerprint feature extraction and performs 1:N open-set identification.**

</div>

---

## 📋 Table of Contents

- [About](#-about)
- [Team Members](#-team-members)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Evaluation](#-evaluation)
---

## 🎯 About

This project implements a fingerprint biometric identification system designed for educational research in Cyber-Physical Systems. It utilizes local feature descriptors (RootSIFT) combined with global intensity histograms to identify individuals from a database of 500 unique identities. The system is designed for **open-set identification**, meaning it can correctly label a probe as "Unknown" if the match quality falls below a tuned threshold.

![Status](https://img.shields.io/badge/Status-Complete-green?style=flat-square)
![Version](https://img.shields.io/badge/Version-1.0.0-blue?style=flat-square)

---

## 👥 Team Members

| Member Name                | Unity ID |
| -------------------------- | -------- |
| Edward Feng                | sfeng9   |
| Jaewoo Bae             | jbae7    |

---

## ✨ Features

- **Advanced Preprocessing**: Uses Median Blur for noise reduction and CLAHE (Contrast Limited Adaptive Histogram Equalization) for ridge enhancement.
- **RootSIFT Descriptors**: Implements Hellinger kernel normalization on SIFT descriptors to significantly improve matching accuracy on textured ridge patterns.
- **Hybrid Scoring**: Fuses local feature matching counts with global intensity histogram cosine similarity for robust verification.
- **Open-Set Support**: Tuned threshold (28.5) and margin (1.6) logic to distinguish between enrolled users and imposters.
- **Comprehensive Analytics**: Automatically generates ROC curves (AUC calculation), Normalized Confusion Matrices, and F1-score metrics.

---

## 🛠️ Tech Stack

- **Core**: Python 3.7+
- **Computer Vision**: OpenCV, Scikit-Image
- **Machine Learning/Stats**: Scikit-Learn, NumPy
- **Visualization**: Matplotlib, Seaborn

---

## 📦 Installation

### Prerequisites

![Requirements](https://img.shields.io/badge/Requirements-Check-green?style=flat-square)

- Python 3.7 or higher

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd fingerprint_authentication-system
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Running the Application
The system will process the data/train folder to enroll users and then evaluate using the data/test folder.

```bash
python main.py
```

### Output Files
After execution, the system generates three primary evaluation files:
- **roc_curve.png**: Shows the trade-off between False Positives and True Positives.
- **confusion_matrix.png**: A normalized matrix for the first 10 identities.
- **evaluation_plot.png**: A summary dashboard of Accuracy, Precision, Recall, and F1-score.

---

## 📁 Project Structure

```
fingerprint-authentication-system/
├── data/
│   ├── train/            # Images used for enrollment (3 per user)
│   ├── test/             # Images used for testing (1 per user)
│   └── validate/         # Validation set
├── src/
│   ├── preprocessing.py  # Image enhancement (CLAHE, Normalization)
│   ├── feature_extraction.py # RootSIFT & Histograms
│   ├── verification.py   # Hybrid scoring & 1:N identification
│   ├── evaluation.py     # ROC and Confusion Matrix logic
│   └── utils.py          # File IO and parsing
├── main.py               # Main pipeline execution
├── requirements.txt      # Project dependencies
├── confusion_matrix.png
├── evaluation_plot.png
├── roc_curve.png
├── 📄 README.md          # This file

```
## 📊 Evaluation

The system currently achieves an AUC (Area Under Curve) of 0.95, indicating excellent separability between genuine fingerprint matches and imposters.

- **Threshold**: 28.5 (Minimum score for identification)
- **Margin**: 1.6 (Difference required between top 1 and top 2 candidates)

---

## 📝 Notes


---

## 📄 License

![License](https://img.shields.io/badge/License-Educational-blue?style=flat-square)

This project is for **educational purposes** (CSC591 007 - Project).

---

## 🙏 Acknowledgments

- **Course**: CSC591 007 - Cyber-Physical Systems for Biometrics
- **Institution**: North Carolina State University
- **Project**: Fingerprint Authentication System

---
