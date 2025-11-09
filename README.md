# 🚁 Acoustic Drone Classification: Memory-Optimized for Resource-Constrained Environments

## 🎯 Project Goal

To develop a high-accuracy, multi-class acoustic classifier capable of distinguishing between **Non-Drone**, **Decoy Drone**, and **Combat Drone** audio signatures. The project's primary constraint and innovation lie in its optimization for **resource-constrained hardware** (e.g., a standard 8GB RAM machine), ensuring high performance without memory explosion.

---

## ✨ Key Technical Innovations (The Memory-First Strategy)

Processing large acoustic datasets is memory-intensive. We implemented a 4-point strategy to maximize accuracy while strictly minimizing peak RAM usage:

### 1. Strategic Data Balancing (Memory-Lite Oversampling)

We tackled a highly imbalanced dataset (Non-Drone: 10k+, Drone classes: ~880). Instead of memory-intensive techniques (like full SMOTE), we used a hybrid approach:
* **Undersampled** the dominant **Non-Drone** class (from 10,373 down to 1,760).
* Applied **Simple Augmentation** (noise, shift, scale) to the minority **Drone** classes to reach the target of **1,760** samples each.
* *Result: A perfectly balanced dataset of 5,280 samples without heavy memory overhead.*

### 2. Enhanced & Compact Feature Engineering

We designed a feature set that is highly discriminative yet compact:
* Features include the raw Mel-filter bank, plus **temporal statistics** (mean, standard deviation, max over time) and **spectral rolloff**.
* This produced a robust, 1593-dimensional feature vector per sample.

### 3. Incremental Processing Pipeline

To manage the memory footprint of feature extraction, especially on a large dataset:
* Feature extraction was performed on the training set in small **batches (500 clips at a time)**.
* Aggressive use of **Python's Garbage Collection (`gc.collect()`)** was implemented to clear processed audio clips and intermediate feature arrays immediately after use, keeping peak RAM low.

### 4. Dimensionality Reduction & Compact Ensemble

* **PCA (Principal Component Analysis):** Applied to reduce the feature vector from 1593 to just **150 components**, retaining **97.25%** of the variance. This greatly optimized model training time and memory footprint.
* **Model Selection:** We trained both a Random Forest and an SVM. We leveraged the high accuracy of the **SVM** (despite its higher memory demand) and the general efficiency of the **Random Forest** in a two-model comparison.

---

## 📈 Performance Summary

The **Support Vector Machine (SVM)** model delivered the highest accuracy and successfully exceeded the target performance threshold.

| Metric | Best Model (SVM) | Ensemble (RF + SVM) |
| :--- | :--- | :--- |
| **Overall Test Accuracy** | **73.11%** | **69.82%** |
| **Variance Explained (PCA)** | 97.25% | - |
| **Target Status** | **✓ Target Accuracy Achieved!** | - |

### Detailed Classification Report (SVM)

| Class | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **Non-Drone** | 71% | 75% | 73% |
| **Decoy Drone** | 71% | **78%** | **75%** |
| **Combat Drone** | **78%** | 67% | 72% |

### Key Findings
* The **Combat Drone** class achieved the highest **Precision (78%)**, demonstrating high reliability when the model identifies this high-priority threat.
* The **Decoy Drone** class achieved the highest **Recall (78%)**, indicating the model is highly effective at detecting this class of drone.

---

## 💻 Technical Details

### Dependencies

This project relies on standard scientific and ML libraries:

```bash
numpy
pandas
scikit-learn
soundfile
matplotlib
