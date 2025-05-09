# CMPE-258 Assignment 8 - Advanced Computer Vision Techniques

This repository contains multiple Google Colab notebooks implementing various advanced computer vision techniques across different modalities, focusing on transfer learning, supervised contrastive learning, zero-shot learning, and state-of-the-art model architectures.

## 📋 Assignment Overview

### Part 1: Supervised Contrastive Learning
Demonstration of supervised contrastive learning loss versus traditional softmax-based classification.

**Key Implementation:**
- Supervised contrastive learning loss function
- Comparison with standard softmax cross-entropy approach
- Performance analysis and visualizations
- Training dynamics comparison

### Part 2: Multi-Modal Transfer Learning
Implementation of transfer learning across various data modalities, both as feature extractors and with fine-tuning.

**Implementations:**
1. **Image Transfer Learning**
   - Dogs vs Cats classification
   - Feature extraction and fine-tuning approaches
   - Performance comparisons with different base models

2. **Video Transfer Learning**
   - Action recognition using pre-trained models
   - Temporal feature extraction
   - Fine-tuning for specific video tasks

3. **Audio Transfer Learning**
   - Using YAMNet for audio classification
   - Transfer learning for custom audio tasks
   - Feature extraction from audio embeddings

4. **NLP Transfer Learning**
   - Text classification using TensorFlow Hub models
   - Fine-tuning language models for specific domains

### Part 3: Zero-Shot Learning with CLIP
Exploration of OpenAI's CLIP model for zero-shot image classification.

**Key Features:**
- Implementation of CLIP for zero-shot classification
- BiT (Big Transfer) model implementation
- Transfer learning with state-of-art TF Hub models
- Performance analysis on unseen classes

### Part 4: Advanced Classifiers and Medical Imaging

**Classifier Implementations:**
1. **Standard Dataset Classifiers**
   - MNIST digit classification
   - Fashion MNIST classification
   - CIFAR-10 image classification
   - Comparison between EfficientNet and BiT transfer learning approaches
   - Implementation of MLP-Mixer and ConvNeXt V2 architectures

2. **Medical Imaging Applications**
   - X-ray pneumonia classification using ConvNets
   - 3D image classification for CT scans
   - Medical-specific preprocessing and augmentation
   - Performance evaluation in medical contexts

## 🔧 Technical Implementations

### Supervised Contrastive Learning
- Implementation of the SupCon loss function
- Two-stage training process (representation learning + linear classification)
- Visualization of learned feature spaces
- Comparison with traditional approaches

### Transfer Learning Techniques
- Feature extraction (freezing pre-trained layers)
- Fine-tuning (updating pre-trained weights)
- Progressive unfreezing strategies
- Learning rate scheduling for optimal transfer

### State-of-the-Art Models
- EfficientNet implementation and fine-tuning
- BiT (Big Transfer) for high-performance transfer learning
- MLP-Mixer architecture for vision tasks
- ConvNeXt V2 implementation

### Medical Imaging Specifics
- Domain-specific preprocessing for X-rays
- 3D convolutions for volumetric medical data
- Handling class imbalance in medical datasets
- Evaluation metrics specific to medical applications

## 📊 Visualizations Included
- t-SNE plots of feature embeddings
- Grad-CAM visualizations for model interpretability
- Training and validation curves
- Confusion matrices for classification performance
- Feature map visualizations
- Attention maps for transformer-based models

## 📹 Video Walkthrough
A comprehensive video walkthrough of all implementations is available at:

[**Watch the Complete Computer Vision Assignment Walkthrough**](https://youtu.be/svYWxBAJI8s)

The video covers:
- Detailed explanation of each implementation
- Code walkthrough and key architectural decisions
- Results analysis and performance comparisons
- Common challenges and solutions
- Best practices for each technique
- Practical applications and future directions

## 🔍 Key Results and Findings
- Supervised contrastive learning outperforms traditional methods on [specific dataset]
- Transfer learning reduces training time by approximately [X%] while improving accuracy
- Zero-shot CLIP model achieves [Y%] accuracy on unseen classes
- BiT models outperform EfficientNet by [Z%] on CIFAR-10
- 3D convolutions improve medical imaging classification accuracy by [W%] compared to 2D approaches

## 🛠️ Dependencies and Setup
All notebooks are designed to run in Google Colab with minimal setup:
- TensorFlow 2.x
- PyTorch (for CLIP implementation)
- TensorFlow Hub
- Scikit-learn
- Matplotlib and Seaborn for visualizations
- Additional libraries as specified in individual notebooks

## 🔗 References
- [Supervised Contrastive Learning Paper](https://arxiv.org/abs/2004.11362)
- [TensorFlow Blog: Transfer Learning for Audio](https://blog.tensorflow.org/2021/03/transfer-learning-for-audio-data-with-yamnet.html)
- [TensorFlow Hub: Action Recognition](https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub)
- [TensorFlow Hub: Text Classification](https://www.tensorflow.org/hub/tutorials/tf2_text_classification)
- [Keras: Supervised Contrastive Learning](https://keras.io/examples/vision/supervised-contrastive-learning)
- [Keras: BiT Implementation](https://keras.io/examples/vision/bit)
- [CLIP: Connecting Text and Images](https://openai.com/research/clip)
- [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601)
- [ConvNeXt V2 Paper](https://arxiv.org/abs/2301.00808)
