# African Wildlife Image Classification Pipeline

## Overview

This repository contains a two-stage transfer learning deep learning pipeline designed to classify the "Big Five" African wildlife species: Buffalo, Elephant, Rhino, Lion, and Leopard. The project evaluates the practical performance impacts of feature extraction versus full network fine-tuning using a pre-trained ResNet50 convolutional neural network.

## Dataset

The dataset was aggregated by merging and filtering relevant classes from the "African Wildlife" and "Big Cats" datasets from Kaggle.

* **Classes:** Buffalo, Elephant, Rhino, Lion, Leopard
* **Pipeline:** Ingested via `tf.data` and standardized to 224x224x3 spatial dimensions.
* **Data Split:** 70% Training / 15% Validation / 15% Testing.
* **Augmentation:** Applied exclusively to the training subset (Horizontal Flip, 10% Random Rotation, 10% Random Zoom) to mitigate overfitting on limited data.

## Architecture

The core architecture relies on **ResNet50** pre-trained on ImageNet. The original 1000-class classification head was truncated (`include_top=False`) and replaced with a custom downstream block:

* `GlobalAveragePooling2D()` (Flattens spatial dimensions mathematically)
* `Dense(256, activation='relu')`
* `Dropout(0.5)` (Enforces structural regularization)
* `Dense(5, activation='softmax')` (Outputs probability distribution for the 5 classes)

## Training Methodology

The training was conducted in two controlled phases to isolate the impact of localized weight adjustments.

### Phase 1: Feature Extraction (Experiment A)

* The entire ResNet50 backbone was frozen.
* Only the custom classification head was trained for 10 epochs using the Adam optimizer (Learning Rate = 1e-3).
* This phase leveraged the generic edge and texture detectors pre-trained on ImageNet.

### Phase 2: Fine-Tuning (Experiment B)

* The top 32 layers of the ResNet50 backbone were unfrozen; the bottom 143 layers remained locked.
* The model was re-compiled with a highly conservative learning rate (1e-5) to prevent catastrophic forgetting.
* The network was trained for an additional 10 epochs to adapt complex shape detectors specifically to African wildlife morphology.

## Results & Evaluation

Fine-tuning the upper layers of the network yielded a measurable performance improvement in peak validation accuracy over baseline feature extraction.

| Metric Evaluated | Experiment A (Feature Extraction) | Experiment B (Fine-Tuning) |
| --- | --- | --- |
| **Best Validation Accuracy** | 96.88% | 97.77% |
| **Test Set Macro F1-Score** | 0.95 | 0.94 |
| **Average Epoch Training Time** | 27.37 sec | 26.56 sec |

**Error Analysis Highlights:**

* **Dominant Class:** The network successfully classified 57 out of 58 elephants (0.98 recall), making it the most robust classification in the pipeline.
* **Primary Vulnerability:** The highest misclassification rate occurred between Buffalo and Rhino due to morphological overlap (heavy build, uniform color). The model incorrectly labeled 5 actual buffaloes as rhinos.

## Usage & Source Code

The complete, reproducible codebase—including data ingestion pipelines, model architecture, training loops, and evaluation metrics—is hosted interactively on Google Colab.

[Open African Wildlife Image Classification Pipeline in Google Colab](https://colab.research.google.com/drive/1iVQnKyv62oc7oNs4WYTj2i75YxJeFY8S?usp=sharing)

