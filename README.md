# 572project
# Multi-Stage IoT Network Intrusion Detection Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A sophisticated two-stage hybrid Intrusion Detection System (IDS) that combines unsupervised anomaly detection with supervised classification for IoT network security.

## 📖 Overview

This project implements a hybrid IDS that:
1. **First Stage**: Uses an autoencoder for unsupervised anomaly detection on packet-level data
2. **Second Stage**: Employs supervised classifiers for precise attack classification on flow-level data
3. **Achieves**: **0% false positive rate** while maintaining high attack detection coverage

## 🎯 Key Features

- **Hybrid Architecture**: Combines unsupervised + supervised learning
- **Real-world Data**: Uses CIC IoT-DIAD 2024 dataset
- **Imbalanced Handling**: 97% normal vs 3% attack traffic distribution
- **Flow Aggregation**: Handles 2-minute flow segments automatically
- **Comprehensive Evaluation**: Full two-stage system performance analysis

## 📊 Results Summary

| Metric | Phase 1 (Autoencoder) | Phase 2 (Classifier) | Final System |
|--------|----------------------|---------------------|-------------|
| **Accuracy** | 92% | 99.34% | 93.5% |
| **False Positive Rate** | 5.08% | - | **0.00%** |
| **Attack Recall** | 50% | - | Maintained |

**Flow Classifier**: 98.25% accuracy with comprehensive attack type analysis

## 🏗️ Project Structure
