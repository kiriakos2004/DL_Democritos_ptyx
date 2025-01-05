# Predicting Ship Propulsion Power using Physics-Informed Neural Networks (PINN)

## Introduction

This project belongs to master's thesis of the Inter-Institutional MSc entitled "Artificial Intelligence" that is organized by The Department of Digital Systems, School of Informatics and Communication Technologies, of University of Piraeus, and the Institute of Informatics and Telecommunications of NCSR "Demokritos". url: "https://msc-ai.iit.demokritos.gr/en". Project explores the use of and Physics-Informed Neural Networks (PINNs) to predict ship propulsion power. By incorporating physical laws into the neural network training process, we aim to improve the model's predictive capabilities and assess whether PINNs offer advantages over purely data-driven models.

## Repository Structure

- read_data_fragment.py: Reads and preprocesses the data, handling missing values, scaling features, and splitting the dataset into training and testing sets. Also it implements a further split of the training dataset in order to reduce it for testing purposes.
- main_COMBINED.py: Implements a Physics-Informed Neural Network that incorporates PDEs related to ship resistance and respective analytical equations related to ship resistance into the training process. Includes hyperparameter tuning and model evaluation. this code 4 coefficients are introduced, each of them corresponds to a different loss term that is summed to form the total loss that is used at the training proceess of the NN.
        - data_loss_coeff: defines the importance of the data loss (if set to 1 and all others to 0, corresponds to a purely data driven model set as baseline)
        - physics_loss_coeff: defines the importance of the physics loss
        - pde_loss_coeff: defines the importance of the partial differential equation loss
        - boundary_loss_coeff: defines the importance of the boundary conditions loss
  
## Features

- Data Preprocessing: Handles missing values and scales features using StandardScaler.
- Data-Driven Model: A multi-layer neural network trained solely on data to predict propulsion power.
- Physics-Informed Neural Network: Enhances the data-driven model by adding a PDE loss term derived from ship resistance partial derivative equations, enhances it by adding a physics-based loss term derived from ship resistance equations and also adds a boundary loss term that "restricts" results based on ships trial data.
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

        python read_data_fragment.py

This will:

- Load the dataset.
- Drop specified columns (e.g., TIME).
- Handle missing values by filling them with the mean.
- Split the data into features and target variable.
- Scale the features using StandardScaler.
- Split the data into training and testing sets.
- Introduce a train_fraction term that takes values from o to 1 and is used to keep a fragment of the initial train dataset.
  
### Check functions used for Physical Loss of PGNN

In order to check if the equations used at the physical part of loss accurately predict power:
        
        physics_model.py
        
 This script will:

- Use unscaled data from data loader to Calculate the power needed.
- Create a csv file logging the calculated power and the power specified on the data in order to compare the allingment.

### Running the Physics-Informed Neural Network (PINN)

To train and evaluate the PINN:

        python main_COMBINED.py

This script:

- Loads and preprocesses the data.
- Incorporates Physical Laws into the Model by defining the governing Partial Differential Equations (PDEs) related to fluid dynamics around the ship hull and computes PDE residuals using automatic differentiation to ensure the model adheres to the underlying physical principles.
- Incorporates physical laws related to ship resistance into the loss function.
- Adds a boundary loss term that penalties the model for producing results that dont allign with ships sea trial tests (make use of extreme values)
- Perform hyperparameter tuning (learning rate and batch size) using k-fold cross-validation.
- Trains the final PINN model with the best hyperparameters.
- Evaluates the PINN on the test set in terms of RMSE.  

## Formulation of a PDE for the Problem

Assuming we can model the resistance R, and thus the required power P, as a function of speed V and other variables, we will consider a PDE like:

![Screenshot_black_letters_white_background](https://github.com/user-attachments/assets/6a7bd284-da46-4581-bdf5-c870e4800bd1)

This PDE express the empirical relationship of ships needed power with the desired speed.

Where:
- P is the Power (the target variable of the data model).
- V is the Speed-Through-Water.

## Physics-Based Loss Function

The PINN incorporates a physics-based loss term calculated using ship resistance equations:

### Frictional Resistance

![image](https://github.com/user-attachments/assets/9a426761-bc2d-4d29-8b45-926fc6892217)

It is calculated using the ITTC-1957 formula, the frictional resistance coefficient accounts for the friction between the ship's hull and the water.

where:

   - k: Form factor accounting for additional resistance due to the ship's shape
   - ρ: Density of water in kg/m3
   - V: Speed in m/s
   - S: Wetted surface area in m2
   - Cf: Frictional resistance coefficient

### Residuary Resistance

![image](https://github.com/user-attachments/assets/56c0a073-780b-4cfc-9211-cac2e5a63b56)

where:

   - Cresid: Residuary resistance coefficient
   - Rfriction: Frictional resistance in N
     
### Wave-Induced Resistance

 ![image](https://github.com/user-attachments/assets/e03f2537-8b2e-4344-9212-ac0277703c09)

where:

   - ρ: Density of water in kg/m3
   - g: Acceleration due to gravity
   - A: Wave amplitude, calculated as:

     ![image](https://github.com/user-attachments/assets/6b6c57f5-0a00-4fbf-a54e-b72a9bb0a0a9)

     where SWH is the significant wave height in m
   - Swave: Wetted surface area affected by waves in m2
   - Cwave: Wave resistance coefficient


Finally the physics-based loss is computed using the total resistance and propulsive efficiency:

![image](https://github.com/user-attachments/assets/48cba486-fed4-4ccf-844c-6110ab7ff0ae)

where:

   - Rtotal: Total resistance in N
   - V: Speed in m/s
   - η: Propulsion efficiency (unitless)


## Boundary Loss

The model is penalized when it produces results that do not adhere to two extreme conditions derived from the ship’s sea trials:

- Condition 1: P ~ 0 for speeds in [0, 1) knots
- Condition 2: P ≥ 8000 kW when V >= 13 knots
  
## Hyperparameter Tuning

Both models perform hyperparameter tuning a predifined Hyperparameter Grid using k-fold cross-validation (default is 5 folds).

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
