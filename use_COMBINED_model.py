import torch
import pandas as pd
import joblib
import numpy as np
import json
from read_data import DataProcessor
from main_COMBINED import ShipSpeedPredictorModel

if __name__ == "__main__":
    # Define file paths and parameters
    model_state_dict_path = 'saved_COMBINED/Adam_relu_Optimizer_dropout_0,2/final_model.pth'
    scaler_X_path = 'saved_COMBINED/Adam_relu_Optimizer_dropout_0,2/scaler_X.save'
    scaler_y_path = 'saved_COMBINED/Adam_relu_Optimizer_dropout_0,2/scaler_y.save'
    test_data_file = 'data/Hera/P data_20220607-20230127_Democritos.csv'  # Update if necessary
    keep_columns_file = 'columns_to_keep.txt'
    target_column = 'Power'
    output_csv_file = 'power_predictions.csv'

    # Load the saved scalers
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)

    # Initialize DataProcessor to load and prepare the test data
    data_processor = DataProcessor(
        file_path=test_data_file,
        target_column=target_column,
        keep_columns_file=keep_columns_file
    )

    # Assign the loaded scalers to the data_processor
    data_processor.scaler_X = scaler_X
    data_processor.scaler_y = scaler_y

    # Load and prepare data (we only need the test set)
    result = data_processor.load_and_prepare_data()
    if result is not None:
        X_train, X_test, X_train_unscaled, X_test_unscaled, y_train, y_test, y_train_unscaled, y_test_unscaled = result

        # Ensure columns are in the same order as during training
        X_test_unscaled = X_test_unscaled[X_train_unscaled.columns]
        X_test = X_test[X_train.columns]

        # Load the best hyperparameters (optional, for consistency)
        with open('saved_COMBINED/Adam_relu_Optimizer_dropout_0,2/best_hyperparameters.json', 'r') as f:
            best_params = json.load(f)

        optimizer = 'Adam'        # Should match what was used during training
        loss_function = 'MSE'     # Should match what was used during training

        # Initialize the model
        loaded_model = ShipSpeedPredictorModel(
            input_size=X_test.shape[1],
            lr=best_params['lr'],
            optimizer_choice=optimizer,
            loss_function_choice=loss_function,
            batch_size=best_params['batch_size'],
            alpha=best_params['alpha'],
            beta=best_params['beta'],
            gamma=best_params.get('gamma', 0.1)  # Include gamma if available, else default to 0.1
        )
        loaded_model.model.load_state_dict(
            torch.load(model_state_dict_path, map_location=loaded_model.device)
        )
        loaded_model.model.to(loaded_model.device)
        loaded_model.model.eval()

        # Prepare the test data
        # Ensure columns are in the same order as during training
        X_test_unscaled = X_test_unscaled[X_train_unscaled.columns]
        X_test_scaled = scaler_X.transform(X_test_unscaled)
        X_test_tensor = torch.tensor(
            X_test_scaled, dtype=torch.float32
        ).to(loaded_model.device)

        # Make predictions
        with torch.no_grad():
            y_pred_scaled = loaded_model.model(X_test_tensor)
            y_pred = data_processor.inverse_transform_y(
                y_pred_scaled.cpu().numpy()
            ).flatten()

        # Get actual power values from the unscaled test target
        y_actual = y_test_unscaled[target_column].values

        # Create a DataFrame with actual and predicted power
        results_df = pd.DataFrame({
            'Actual Power': y_actual,
            'Predicted Power': y_pred
        })

        # Save the results to a CSV file
        results_df.to_csv(output_csv_file, index=False)
        print(f"Predictions saved to {output_csv_file}")

        # Optionally, calculate and print evaluation metrics
        rmse = np.sqrt(np.mean((y_actual - y_pred) ** 2))
        mae = np.mean(np.abs(y_actual - y_pred))
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test MAE: {mae:.4f}")
    else:
        print("Error in loading and preparing data.")
