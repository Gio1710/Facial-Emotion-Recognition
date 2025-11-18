# Facial-Emotion-Recognition

This repository contains the implementation and analysis of a Deep Learning model designed to classify human facial expressions into five categories: **Angry, Fear, Happy, Sad, and Surprise**.

The project follows a rigorous data science methodology, moving from a custom CNN baseline to a sophisticated Transfer Learning approach, achieving **91% accuracy** on external test data.

## 📊 Results Overview

Through iterative experimentation, the final model (V13) achieved state-of-the-art performance by leveraging:
1.  **ResNet50V2 Architecture** (Transfer Learning)
2.  **Class Weighting** (to handle dataset imbalance)
3.  **Fine-Tuning** of the upper layers

| Metric | Internal Validation | External Test (FER-2013) |
| :--- | :---: | :---: |
| **Accuracy** | **87%** | **91%** |
| F1-Score (Fear) | 0.81 | 0.84 |
| F1-Score (Happy) | 0.92 | 0.95 |

## 📂 Project Structure

- `src/train_model.py`: The complete Python pipeline for data loading, preprocessing, model training (ResNet50V2), and evaluation.
- `report/`: Contains the detailed LaTeX scientific report and analysis plots.
- `requirements.txt`: List of dependencies to run the project.

## 🚀 Methodology Highlights

- **Data Analysis:** Identified and corrected severe class imbalance using computed `class_weights`.
- **Failed Experiments:** Documented the negative impact of geometric and photometric Data Augmentation for this specific task.
- **Architecture Search:** Compared MobileNetV2, EfficientNetB0, and ResNet50V2, with the latter proving superior.

## 🛠️ How to Run

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR-USERNAME/facial-emotion-recognition.git](https://github.com/YOUR-USERNAME/facial-emotion-recognition.git)
   cd facial-emotion-recognition
