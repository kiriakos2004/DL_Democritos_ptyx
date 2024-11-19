import torch
import pandas as pd
import joblib
import numpy as np
from read_data import DataProcessor
from main_PINN import ShipSpeedPredictorModel  # Import the model class from your main script

if __name__ == "__main__":
    # Define file paths and parameters
    model_state_dict_path = 'saved_PINN/LBFGS_Optimizer/final_model.pth'
    scaler_X_path = 'saved_PINN/LBFGS_Optimizer/scaler_X.save'
    scaler_y_path = 'saved_PINN/LBFGS_Optimizer/scaler_y.save'
    test_data_file = 'data/Aframax/P data_20200213-20200726_Democritos.csv'  # Update if necessary
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

    # Load and prepare data (we only need the test set)
    result = data_processor.load_and_prepare_data()
    if result is not None:
        X_train, X_test, X_train_unscaled, X_test_unscaled, y_train, y_test, y_train_unscaled, y_test_unscaled = result

        # Load the best hyperparameters (optional, for consistency)
        import json
        with open('saved_PINN/LBFGS_Optimizer/best_hyperparameters.json', 'r') as f:
            best_params = json.load(f)

        optimizer = 'LBFGS'        # Should match what was used during training
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
            gamma=best_params['gamma'],
            debug_mode=False
        )
        loaded_model.model.load_state_dict(torch.load(model_state_dict_path, map_location=loaded_model.device))
        loaded_model.model.eval()

        # Prepare the test data
        # Ensure columns are in the same order as during training
        X_test_unscaled = X_test_unscaled[X_train_unscaled.columns]
        X_test_scaled = scaler_X.transform(X_test_unscaled)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(loaded_model.device)

        # Make predictions
        with torch.no_grad():
            y_pred_scaled = loaded_model.model(X_test_tensor)
            y_pred = scaler_y.inverse_transform(y_pred_scaled.cpu().numpy()).flatten()

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