---
title: MLOps and AWS projects
url: https://github.com/Data-Fenix/aws-sagemaker-pipeline
---

## Customer churn prediction: end-to-end AWS SageMaker pipeline

Repository: https://github.com/Data-Fenix/aws-sagemaker-pipeline. An ML workflow is a sequence of ML tasks that run in order; this project demonstrates a complete end-to-end pipeline on AWS SageMaker Pipelines for a telco customer churn use case. Identifying churning customers saves the telecom company millions. The pipeline covers data ingestion, preprocessing, XGBoost training, evaluation, conditional model registration and deployment. It uses the "bring your own code" concept: custom scripts are containerised with Docker and the inference job runs a Flask app. The dataset has 7,043 telco customers with 21 features and a binary Churn target.

## Scheduled training and inference jobs for churn

Repositories: https://github.com/Data-Fenix/aws-sagemaker-training-job-customer-churn-prediction and https://github.com/Data-Fenix/aws-sagemaker-inference-job-customer-churn-prediction. When a client needs predictions every week, running the full pipeline each time wastes time and money. The workflow is split into a training job (preprocessing and model training, run quarterly) and an inference job (preprocessing and inferencing, run daily or weekly). The scheduling cadence is adjusted to the organisation's requirements and cost constraints.

## Conditional step SageMaker training pipeline

Repository: https://github.com/Data-Fenix/condition-step-aws-sagemaker-training-pipeline. Demonstrates a SageMaker pipeline with a condition step that registers a model only when its evaluation metric passes a threshold.

## Mobile price prediction SageMaker training job

Repository: https://github.com/Data-Fenix/aws-sagemaker-training-job-mobile-price-prediction. Runs the mobile price regression model as a SageMaker training job.

## Maintaining an AWS CodeCommit repository

Repository: https://github.com/Data-Fenix/maintain-aws-codecommit-repository. A guideline project on how to structure and maintain an AWS CodeCommit repository for ML teams.

## DevOps standardisation and Git workflows

Repositories: https://github.com/Data-Fenix/gitworkflows, https://github.com/Data-Fenix/DevOps and https://github.com/Data-Fenix/Testing. Shell-based Git hooks, GitHub Actions workflows and standardised environments used to streamline engineering operations. The Testing repository exercises Git hooks and GitHub Actions.
