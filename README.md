# CMPE-258-Assignment-8-Computer-vision-assignment

```markdown
# Assignment 8 - Computer Vision

This repository contains solutions to Assignment 8 for the Computer Vision course. The assignment is structured into multiple parts covering contrastive learning, transfer learning, and the application of state-of-the-art models for vision tasks.

---

## 📁 Structure

```
assignment_8/
│
├── Part1_SupervisedContrastiveLearning/
│   ├── supervised_contrastive_vs_softmax.ipynb
│   └── README.md
│
├── Part2_TransferLearning_Modalities/
│   ├── image_transfer_learning.ipynb
│   ├── audio_transfer_learning.ipynb
│   ├── video_transfer_learning.ipynb
│   └── README.md
│
├── Part3_ZeroShot_and_TFHub/
│   ├── zero_shot_CLIP.ipynb
│   ├── tfhub_transferlearning_bigtransfer.ipynb
│   └── README.md
│
├── Part4_VisionModels/
│   ├── mnist_transferlearning.ipynb
│   ├── fashion_mnist_transferlearning.ipynb
│   ├── cifar10_transferlearning.ipynb
│   └── README.md
│
├── Part5_MedicalImaging/
│   ├── xray_pneumonia_classification.ipynb
│   ├── ct_scan_3d_classification.ipynb
│   └── README.md
│
└── README.md  ← (This file)
```

---

## ✅ Part 1: Supervised Contrastive Learning vs Softmax

- Demonstrates the use of **Supervised Contrastive Loss** vs traditional **Softmax Cross-Entropy** for image classification.
- Dataset: CIFAR-10
- Visualizes learned embeddings using t-SNE/UMAP.
- Reference: [Keras example](https://keras.io/examples/vision/supervised-contrastive-learning/)

---

## ✅ Part 2: Transfer Learning on Modalities (Image, Video, Audio)

Includes transfer learning examples for different input modalities:

- **Images:** Cats vs Dogs or Dog Breed Classification
  - Using both feature extraction and fine-tuning.
- **Audio:** Using YAMNet for classifying urban sounds or similar.
  - [YAMNet Tutorial](https://blog.tensorflow.org/2021/03/transfer-learning-for-audio-data-with-yamnet.html)
- **Video:** Action recognition using TFHub I3D features.
  - [Video Action Recognition](https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub)

---

## ✅ Part 3: Zero-Shot Transfer Learning & TFHub (SOTA)

- **Zero-Shot Learning with CLIP** (Contrastive Language-Image Pretraining)
  - Classify images without task-specific training.
  - [CLIP Tutorial](https://towardsdatascience.com/how-to-try-clip-openais-zero-shot-image-classifier-439d75a34d6b)
- **Transfer Learning with BigTransfer (BiT)**
  - Fine-tuned on flower datasets using BiT from TFHub.

---

## ✅ Part 4: Vision Classification on Standard Datasets

Three notebooks applying transfer learning with EfficientNet and BiT models on:

- MNIST
- Fashion MNIST
- CIFAR-10

Each notebook includes:
- Feature extraction
- Fine-tuning
- SOTA architecture comparisons (ConvNeXt V2, MLP-Mixer)

---

## ✅ Part 5: Medical Imaging Use Cases

- **X-ray Pneumonia Classification:**
  - CNN-based classification on chest X-ray images.
  - Includes data preprocessing and augmentation.
- **3D CT Scan Classification:**
  - Uses volumetric data for binary classification (e.g. presence of tumor).
  - Based on [3D image classification](https://keras.io/examples/vision/3D_image_classification/)

---

## 🔧 Requirements

Install required packages using:

```bash
pip install tensorflow tensorflow-hub matplotlib seaborn scikit-learn umap-learn
```

All notebooks are Colab-ready and make use of GPU/TPU where appropriate.

---

## 📊 Visualizations

- Embedding visualizations using **t-SNE** and **UMAP**.
- Training curves comparing different methods.
- Activation maps (for X-ray and CT Scan models).

---

## 📚 References

- [Keras Vision Examples](https://keras.io/examples/vision/)
- [TensorFlow Hub Tutorials](https://www.tensorflow.org/hub)
- [Contrastive Loss Guide](https://towardsdatascience.com/contrastive-loss-for-supervised-classification-224ae35692e7)
- [CLIP Overview](https://github.com/openai/CLIP)
