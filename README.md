# Predicting Ship Propulsion Power using Physics-Informed Neural Networks (PINNs)

## Introduction

This project belongs to master's thesis of the Inter-Institutional MSc entitled "Artificial Intelligence" that is organized by The Department of Digital Systems, School of Informatics and Communication Technologies, of University of Piraeus, and the Institute of Informatics and Telecommunications of NCSR "Demokritos". url: "https://msc-ai.iit.demokritos.gr/en".Project explores the use of Physics-Informed Neural Networks (PINNs) to predict ship propulsion power. By incorporating physical laws into the neural network training process, we aim to improve the model's predictive capabilities and assess whether PINNs offer advantages over purely data-driven models.

## Repository Structure

- read_data.py: Reads and preprocesses the data, handling missing values, scaling features, and splitting the dataset into training and testing sets.
- main_no_pinn.py: Implements a purely data-driven neural network model for predicting ship propulsion power. Includes hyperparameter tuning and model evaluation.
- main_pinn.py: Implements a Physics-Informed Neural Network (PINN) that incorporates physical laws related to ship resistance into the training process.

Features

- Data Preprocessing: Handles missing values and scales features using StandardScaler.
- Data-Driven Model: A multi-layer neural network trained solely on data to predict propulsion power.
- Physics-Informed Neural Network: Enhances the data-driven model by adding a physics-based loss term derived from ship resistance equations.
- Hyperparameter Tuning: Uses grid search and k-fold cross-validation to find optimal learning rates and batch sizes.
- Model Evaluation: Provides training and validation loss during training and evaluates the final model on a test set.

The code has been created with the use of python version 3.12.2. In order to recreate the same working enviroment (and to ensure trouble-free code execusion) it is advised to run under virtual enviroment that should be created with the use of requirements.txt (attached).

