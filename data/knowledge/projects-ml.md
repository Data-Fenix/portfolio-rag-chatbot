---
title: Machine learning and analytics projects
url: https://github.com/Data-Fenix?tab=repositories
---

## Smart car parking: predictive availability system

Repository: https://github.com/Data-Fenix/smart-car-parking. A machine learning system that predicts parking spot availability per city road segment from historical occupancy, weather and location data. A Random Forest regressor reaches an R2 of 0.916 with a mean absolute error of 1.04 spots and an RMSE of 2.11 spots. The dataset covers 4 months, 57 road segments, more than 1.25 million observations and 13.4 million parking slots tracked between 06:00 and 21:00. Key insights: peak demand on weekdays 10:00 to 15:00 and 18:00 to 21:00; occupancy drops in extreme temperatures (below 7 C or above 30 C) and rain; roads near restaurants, shopping and residential zones show 40%+ higher occupancy; only 25% of segments have public transport access. Features include temporal, weather, location and engineered lag and rolling features. Delivered as a case study notebook, PDF report and presentation.

## PC and electronics price tracker

Repository: https://github.com/Data-Fenix/pc-price-tracker. A daily price scraper for laptops, GPUs and phones across Amazon DE, eBay DE and Idealo DE, storing output in Azure Blob Storage with a date-partitioned layout (raw/{source}/{category}/{year}/{month}/{day}/products.json). Phase 2 cleans the raw records and matches each listing to a product catalogue using NLP, producing a unified price table. Three matching strategies were benchmarked on 135 positives and 20 accessory negatives: rule-based token overlap (precision 0.822, F1 0.902), RapidFuzz token_set_ratio (precision 0.800, F1 0.889) and Sentence-Transformers all-MiniLM-L6-v2 cosine similarity (precision 0.837, F1 0.911, 16.7 ms per record). The Sentence-Transformers matcher performed best, with the highest F1 of 0.911. All matchers reach recall 1.0 and a 0.0 false positive rate thanks to a shared German and English accessory denylist. The catalogue holds 125 synthetic entries (40 laptops, 40 GPUs, 45 phones).

## Potato leaf disease classification

Repository: https://github.com/Data-Fenix/Potato-Leaf-Disease-Classification. A convolutional neural network classifies potato leaf images into Healthy, Early Blight and Late Blight using the PlantVillage dataset from Kaggle, helping farmers detect disease early and reduce crop loss. Steps: data collection and preprocessing, CNN training, evaluation and deployment for inference, with Docker and AWS deployment options.

## Anomaly detection

Repository: https://github.com/Data-Fenix/anomaly-detection. Advanced ML models to detect operational anomalies and surface actionable signals for business decision-making.

## Spam detection

Repository: https://github.com/Data-Fenix/spam-detection. A text classification pipeline for SMS and email spam detection.

## How do we earn more than 50K per year

Repository: https://github.com/Data-Fenix/how-do-we-earn-more-than-50K-per-year. Adult income classification built with a deployable training and inference split, following the same separation of training and inference jobs as the churn project.

## Hotel review sentiment analysis

Repository: https://github.com/Data-Fenix/Are-you-looking-for-best-hotel-to-stay. Sentiment analysis (opinion mining) on hotel reviews. Each observation is one customer review with text and an overall rating; the model classifies polarity as positive, negative or neutral to extract quality signals that guide hospitality decisions.

## Mall customer segmentation

Repository: https://github.com/Data-Fenix/Need-to-Provide-Discounts-and-Promotions-to-the-Customers. K-means clustering on mall customer data to segment customers who share similar characteristics so discounts and promotions can be targeted.

## Mobile price prediction

Repository: https://github.com/Data-Fenix/mobile-price-prediction. Predicts phone prices from brand and features in a web-scraped dataset. Trained linear regression, elastic net, extra trees, random forest and support vector regression, then tuned the best model with RandomizedSearchCV and GridSearchCV and saved it as a pickle for future predictions.

## Electricity price prediction

Repository: https://github.com/Data-Fenix/electricity_price_prediction. Regression models for forecasting electricity prices.

## Hawaii real estate and Airbnb analysis

Repositories: https://github.com/Data-Fenix/best-place-to-live-in-Hawaii and https://github.com/Data-Fenix/airbnb-data-analysis-hawaii. Exploratory data analysis of the Hawaii real estate market and of Airbnb listings in Hawaii.

## pedata: protein engineering dataset handling

Repository: https://github.com/Data-Fenix/pedata. A Python library for handling protein engineering datasets, with the black formatter enforced, pdoc documentation and registered Git hooks.
