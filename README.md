# ASU CEN 598: Embedded Machine Learning (TinyML) Projects

This repository contains a series of projects developed for the **Embedded Machine Learning** course at Arizona State University. The projects demonstrate the implementation of machine learning workflows on resource-constrained hardware.

## Tech Stack & Tools
* **Hardware**: Arduino Nano 33 BLE Sense Rev2.
* **Frameworks**: TensorFlow Lite for Microcontrollers (TFLite Micro), PyTorch, and Keras.
* **Protocols**: Bluetooth Low Energy (BLE) for real-time communication and control.
* **Sensors**: IMU (Inertial Measurement Unit) and PDM Microphone.

---

## Projects Overview

### Food Image Classification
This project investigates image classification using CNN-based models on the Food-11 dataset. It explores the transition from basic convolutional layers to lightweight architectures like ShuffleNet V2, focusing on improving generalization and reducing overfitting for deployment on mobile or Arduino platforms.

### Real-Time Keyword Spotting (KWS)
The objective is to design an end-to-end audio classification system capable of recognizing specific keywords ("Never", "None", "All", "Must", "Only"). The workflow includes custom data collection, audio feature extraction into spectrograms, and deploying an INT8-quantized model for real-time inference with LED feedback.

### Sensor-Agnostic Posture Detection
This project focuses on building a posture detection system that can predict static positions regardless of the specific sensor used. By utilizing a 1D CNN and a per-sensor normalization strategy, the system can classify postures using data from either an accelerometer, gyroscope, or magnetometer.

### Multi-Class Posture Tracking & Analysis
An extension of lying posture tracking that incorporates hyperparameter tuning and model optimization. This project involves a comparative
