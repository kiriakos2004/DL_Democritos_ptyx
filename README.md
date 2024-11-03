# Predicting Ship Propulsion Power using Physics-Informed Neural Networks (PINNs)

## Introduction

This project belongs to master's thesis of the Inter-Institutional MSc entitled "Artificial Intelligence" that is organized by The Department of Digital Systems, School of Informatics and Communication Technologies, of University of Piraeus, and the Institute of Informatics and Telecommunications of NCSR "Demokritos". url: "https://msc-ai.iit.demokritos.gr/en".Project explores the use of Physics-Informed Neural Networks (PINNs) to predict ship propulsion power. By incorporating physical laws into the neural network training process, we aim to improve the model's predictive capabilities and assess whether PINNs offer advantages over purely data-driven models.

## Repository Structure

- read_data.py: Reads and preprocesses the data, handling missing values, scaling features, and splitting the dataset into training and testing sets.
- main_no_pinn.py: Implements a purely data-driven neural network model for predicting ship propulsion power. Includes hyperparameter tuning and model evaluation.
- main_pinn.py: Implements a Physics-Informed Neural Network (PINN) that incorporates physical laws related to ship resistance into the training process.

## Features

- Data Preprocessing: Handles missing values and scales features using StandardScaler.
- Data-Driven Model: A multi-layer neural network trained solely on data to predict propulsion power.
- Physics-Informed Neural Network: Enhances the data-driven model by adding a physics-based loss term derived from ship resistance equations.
- Hyperparameter Tuning: Uses grid search and k-fold cross-validation to find optimal learning rates and batch sizes.
- Model Evaluation: Provides training and validation loss during training and evaluates the final model on a test set.

## Requirements

The code has been created with the use of python version 3.12.2. In order to recreate the same working enviroment (and to ensure trouble-free code execusion) it is advised to run under virtual enviroment that should be created with the use of requirements.txt (attached).

## Installation

1. Clone the repository:

        git clone https://github.com/kiriakos2004/DL_Democritos_ptyx.git

2. Set up a virtual environment (optional but recommended):

        python -m venv <name you want>
        source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. Install dependencies using:

        pip install -r requirements.txt


## Data Preparation

Ensure that you have the required CSV data files in the data/ directory. Update the file paths in the scripts if necessary.

Update the file_path variable in the scripts if your data is located elsewhere.

## Usage
### Data Preprocessing

Before running the models, read and preprocess the data:

        python read_data.py

This will:

- Load the dataset.
- Drop specified columns (e.g., TIME).
- Handle missing values by filling them with the mean.
- Split the data into features and target variable.
- Scale the features using StandardScaler.
- Split the data into training and testing sets.

### Check functions used for Physical Loss of PINN

In order to check if the equations used at the physical part of loss accurately predict power:
        
        power_charts.py
        
 This script will:

- Use unscaled data from data loader to Calculate the power needed.
- Display in a common diagramm the calculated power and the power specified on the data in order to visualy compare the allingment.

### Running the Data-Driven Model

To train and evaluate the purely data-driven neural network model:
        
        python main_no_pinn.py

This script will:

- Load and preprocess the data using DataProcessor.
- Perform hyperparameter tuning (learning rate and batch size) using k-fold cross-validation.
- Train the final model with the best hyperparameters.
- Evaluate the model on the test set.

### Running the Physics-Informed Neural Network (PINN)

To train and evaluate the PINN:

        python main_pinn.py

This script:

- Loads and preprocesses the data.
- Incorporates physical laws related to ship resistance into the loss function.
- Performs hyperparameter tuning similar to the data-driven model.
- Trains the final PINN model with the best hyperparameters.
- Evaluates the PINN on the test set.

## Physics-Based Loss Function

The PINN incorporates a physics-based loss term calculated using ship resistance equations:

- Frictional Resistance
- Wave-Making Resistance
- Appendage Resistance
- Transom Stern Resistance
- Correlation Allowance Resistance

The physics-based loss is computed as the squared difference between the predicted power and the power calculated using the total resistance and propulsive efficiency:

![Screenshot_2](https://github.com/user-attachments/assets/88e88f14-73b3-4232-92ed-693ce98a8c87)

## Hyperparameter Tuning

Both models perform hyperparameter tuning over:

- Learning Rate: [0.001, 0.01]
- Batch Size: [32, 64]

Using k-fold cross-validation (default is 5 folds).

## Results

- Training Loss: Displayed during each epoch for both models.
- Validation Loss: Reported during hyperparameter tuning for each parameter combination.
- Test Loss: Final evaluation metric on the test set.

Compare the test losses of both models to assess the impact of incorporating physics into the model.

## Acknowledgments

- Special thanks to my profeccor Christoforos Rekatsinas (Ph.D.) for his guidance and support.

## Contact

For any questions or inquiries, please contact:

- Alexiou Kiriakos
- Email: kiriakosal2004@yahoo.gr
