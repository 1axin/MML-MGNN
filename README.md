# MML-MGNN: Multi-Channel Graph Neural Network for Cancer Biomarker Prediction

## Overview

MML-MGNN (Multi-Channel Graph Neural Network based on Multi-Similarity Modality Hypergraph Contrastive Learning) is a sophisticated framework designed for predicting unknown cancer biomarkers. By integrating multi-similarity modalities and hypergraph contrastive learning, MML-MGNN enhances the extraction and utilization of node features, leading to more accurate and insightful biomarker predictions.

## Requirements

Ensure the following dependencies are installed:

- **Python**: 3.7
- **Numpy**: 1.21.5
- **Scikit-learn**: 1.2
- **Torch**: 1.10.2
- **Scipy**: 1.7.3

You can install the required packages using pip:

```bash
pip install numpy==1.21.5 scikit-learn==1.2 torch==1.10.2 scipy==1.7.3
```

## Directory Structure

- **MML-MGNN/**: Main directory containing the implementation code.
  -**Comparison results between MML-MGNN and SOTA/**:Prediction scores compared to SOTA models
  -**DATA/**:User-defined data form example
  -**MML-MGNN/**
    - **main_experiments.py**: Runs the MHCL feature extraction process and generates node features.
    - **hypergraph_construct_kmeans.py**: Constructs hypergraphs using KMeans clustering.
    - **hypergraph_construct_KNN.py**: Constructs hypergraphs using KNN clustering.
    - **prepare_data.py**: Constructs and merges kernel similarity modalities.
    - **Module_demo.py**: Defines the MHCL model.
    - **MCGAE.py**: Contains the Multi-Channel Graph Autoencoder implementation for classification and prediction tasks.


## Data Preparation

Prepare your data to include the following similarity modalities:

1. **Sigmoid**: Kernel similarity.
2. **ST**: Structural similarity.
3. **Walker**: Nearest neighbor structural similarity.

Use the `prepare_data.py` script to build and merge these kernel similarity modalities into a format suitable for the MML-MGNN framework.

## Model Components

### Multi-Similarity Modality Hypergraph Contrastive Learning (MHCL)

- **Module_demo.py**: Implements the MHCL model, which performs contrastive learning based on multi-similarity modalities.
- **Feature Extraction**: Execute `main_experiments.py` to extract features using the MHCL approach.

### Multi-Channel Graph Autoencoder (MCGAE)

- **Training and Prediction**: The `MCGAE/` folder contains scripts to train and predict using the features generated by MHCL. The Multi-Channel Graph Autoencoder performs high-level classification tasks and outputs the final prediction results.

## Running the Experiments
1. **Data preparation**:
   - Make sure you prepare your data as per the example
3. **Feature Extraction with MHCL**:
   - Run `main_experiments.py` to perform feature extraction and generate node features using MHCL.
4. **Train and Predict with MCGAE**:
   - Navigate to the `MCGAE.py` folder and run the appropriate scripts to train the model and make predictions based on the generated features.

## Evaluation

The results comparing MML-MGNN with SOTA models are available in the **Comparison results between MML-MGNN and SOTA/** folder. The directory includes the prediction scores of MML-MGNN and SOTA models to facilitate the comparison of the performance of MML-MGNN with state-of-the-art methods.

## Additional Notes

- Ensure that all dependencies are installed as specified.
- Node features should be generated and saved before running the MCGAE training and prediction scripts.

---

For any questions or further assistance, feel free to contact us.
