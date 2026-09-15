# 🛰️ Deep Anomaly Detection using Autoencoders

This project implements a **Deep Convolutional Autoencoder (CAE)** in PyTorch to detect anomalies in physical detector simulation datasets. The model is trained exclusively on normal background data to reconstruct "clean" samples, utilizing reconstruction errors (MSE) to flag outlier patterns.

## 👥 Authors
*   **Aitor Mentxaka** (aitor.mentxaka@estudiante.uam.es - aitormentxaka@hotmail.es)
*   **Miguel Rascón** (miguel.rascon@estudiante.uam.es - mihuit08@gmail.com)
*   **Marcos de Miguel** (marcos.demiguelv@estudiante.uam.es)
*   *Sapienza Università di Roma* (2024 - 2025)

---

## 📊 Datasets
*   `Normal_data.npz`: 12,000 samples of background (normal) detector signals (100x100 grayscale images).
*   `Test_data_low.npz`: Test set containing a low mixture of anomaly signals (<45%).
*   `Test_data_high.npz`: Test set containing a high mixture of anomaly signals (>55%).

---

## 🧠 Model Architecture

### **Encoder**
*   **Conv2d 1**: $1 \rightarrow 32$ channels, kernel $3\times3$, stride $2$
*   **Conv2d 2**: $32 \rightarrow 64$ channels, kernel $3\times3$, stride $2$
*   **Conv2d 3**: $64 \rightarrow 96$ channels, kernel $3\times3$, stride $2$
*   **Fully Connected (fc_z)**: Flat features $\rightarrow 64$-dimensional Latent Space

### **Decoder**
*   **Fully Connected**: $64 \rightarrow 11616$ dimensions (reshaped to $96 \times 11 \times 11$)
*   **ConvTranspose2d 3**: $96 \rightarrow 64$ channels, kernel $3\times3$, stride $2$, output padding $1$
*   **ConvTranspose2d 2**: $64 \rightarrow 32$ channels, kernel $3\times3$, stride $2$
*   **ConvTranspose2d 1**: $32 \rightarrow 1$ channel, kernel $3\times3$, stride $2$, output padding $1$

---

## 📈 Key Findings
*   **Optimal Threshold (10% FPR):** Achieves **59.53%** anomaly detection rate on the *High Anomaly* dataset, and **11.43%** on the *Low Anomaly* dataset.
*   **Dimensionality Reduction:** While PCA and UMAP help visualize global structures, density-based latent clustering (K-Means/GMM) proves limited due to high overlap in the lower-dimensional manifold. Relying on reconstruction error (MSE) yields the most robust separation.