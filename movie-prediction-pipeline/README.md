# Movie Box Office Prediction MLOps Pipeline

A comprehensive MLOps pipeline for predicting movie box office revenue using Amazon SageMaker. This project converts a Jupyter notebook analysis into a production-ready, modular machine learning pipeline with automated training, evaluation, and deployment capabilities.

## Project Overview

This pipeline predicts movie box office revenue based on traditional movie features (genre, rating, budget, star power) and internet buzz variables (social media metrics). The original analysis showed that internet buzz variables significantly improve prediction accuracy.

### Key Features

- **Modular Architecture**: Separate modules for data preprocessing, feature engineering, model training, evaluation, and inference
- **SageMaker Integration**: Full MLOps pipeline using Amazon SageMaker Pipelines
- **Advanced Feature Engineering**: Principal Component Analysis (PCA) on buzz variables
- **Model Evaluation**: Comprehensive evaluation with statistical significance testing
- **Automated Deployment**: Conditional model registration based on performance thresholds
- **Scalable Inference**: Batch and real-time prediction capabilities

### Prerequisites

- Python 3.8+
- AWS Account with SageMaker access
- AWS CLI configured
- Required permissions for SageMaker, S3, and IAM

### SageMaker Pipeline Deployment

#### Create and Deploy Pipeline
```bash
python scripts/sagemaker_pipeline.py \
    --config-path config/config.yaml \
    --deploy \
    --execute
```

