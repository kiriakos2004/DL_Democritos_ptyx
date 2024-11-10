import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    def __init__(
        self, file_path, target_column, keep_columns_file, test_size=0.2, random_state=42,
        fill_missing_with_median=True, exclude_missing_Hs=True
    ):
        self.file_path = file_path
        self.target_column = target_column
        self.keep_columns_file = keep_columns_file  # New parameter: path to the text file
        self.test_size = test_size
        self.random_state = random_state
        self.fill_missing_with_median = fill_missing_with_median  # Existing parameter
        self.exclude_missing_Hs = exclude_missing_Hs  # Existing parameter
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.df = None  # Initialize the DataFrame attribute

    def load_and_prepare_data(self):
        # Load CSV file into a DataFrame with exception handling
        try:
            self.df = pd.read_csv(self.file_path)
        except FileNotFoundError:
            print(f"File not found: {self.file_path}")
            return None
        except Exception as e:
            print(f"An error occurred while reading the file: {e}")
            return None

        # Read the columns to keep from the text file
        try:
            with open(self.keep_columns_file, 'r') as f:
                columns_to_keep = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"Columns file not found: {self.keep_columns_file}")
            return None
        except Exception as e:
            print(f"An error occurred while reading the columns file: {e}")
            return None

        # Ensure the target column is included in the columns to keep
        if self.target_column not in columns_to_keep:
            columns_to_keep.append(self.target_column)

        # Filter the DataFrame to keep only the specified columns
        missing_columns = [col for col in columns_to_keep if col not in self.df.columns]
        if missing_columns:
            print(f"Error: The following columns specified in the columns file are not in the dataset: {missing_columns}")
            return None

        self.df = self.df[columns_to_keep]

        # Proceed with data preparation as before
        # Ensure the target column exists
        if self.target_column not in self.df.columns:
            print(f"Error: Target column '{self.target_column}' not found in the dataset.")
            return None

        # Optionally exclude rows with missing H_s values
        if self.exclude_missing_Hs:
            if 'Significant_Wave_Height' in self.df.columns:
                initial_row_count = len(self.df)
                self.df = self.df.dropna(subset=['Significant_Wave_Height'])
                rows_dropped = initial_row_count - len(self.df)
                print(f"Dropped {rows_dropped} rows due to missing 'Significant_Wave_Height'")
            else:
                print("Warning: 'Significant_Wave_Height' column not found in the dataset.")

        # Drop rows where 'Power' is less than 1000
        self.df = self.df[self.df['Power'] >= 1000]

        # Drop rows where 'Speed-Through-Water' is less than 4 knots
        if 'Speed-Through-Water' in self.df.columns:
            self.df = self.df[self.df['Speed-Through-Water'] >= 4]
        else:
            print("Warning: 'Speed-Through-Water' column not found in the dataset.")

        # Check for missing values and fill them
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns

        # Fill missing values with median or mean
        if self.fill_missing_with_median:
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        else:
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())

        # Split the data into features (X) and target (y)
        X = self.df.drop(self.target_column, axis=1)  # Features
        y = self.df[[self.target_column]]  # Target (as DataFrame for scaling)

        # Keep a copy of unscaled features and target
        X_unscaled = X.copy()
        y_unscaled = y.copy()

        # Split the dataset into training and testing sets without shuffling
        X_train_unscaled, X_test_unscaled, y_train_unscaled, y_test_unscaled = train_test_split(
            X_unscaled, y_unscaled, test_size=self.test_size, random_state=self.random_state, shuffle=False
        )

        # Scale the features using the training data
        self.scaler_X.fit(X_train_unscaled)
        X_train_scaled = self.scaler_X.transform(X_train_unscaled)
        X_test_scaled = self.scaler_X.transform(X_test_unscaled)

        # Scale the target variable using the training data
        self.scaler_y.fit(y_train_unscaled)
        y_train_scaled = self.scaler_y.transform(y_train_unscaled)
        y_test_scaled = self.scaler_y.transform(y_test_unscaled)

        # Convert scaled features back to DataFrames to maintain column names
        X_train = pd.DataFrame(X_train_scaled, columns=X_unscaled.columns)
        X_test = pd.DataFrame(X_test_scaled, columns=X_unscaled.columns)

        # Convert scaled targets to Series
        y_train = pd.Series(y_train_scaled.flatten(), name=self.target_column)
        y_test = pd.Series(y_test_scaled.flatten(), name=self.target_column)

        return (
            X_train,
            X_test,
            X_train_unscaled,
            X_test_unscaled,
            y_train,
            y_test,
            y_train_unscaled,
            y_test_unscaled,
        )

    def print_dataset_shapes(self, X_train, X_test):
        # Print the shape of the datasets
        print(f"Training features shape: {X_train.shape}")
        print(f"Test features shape: {X_test.shape}")

    def print_dataset_head(self, X_train, X_test):
        # Print the first few rows of X_train and X_test
        print("First few rows of training features (X_train):")
        print(X_train.head())

        print("\nFirst few rows of test features (X_test):")
        print(X_test.head())

    def list_column_names(self):
        # Ensure the DataFrame is loaded
        if self.df is None:
            try:
                self.df = pd.read_csv(self.file_path)
                # Read the columns to keep from the text file
                with open(self.keep_columns_file, 'r') as f:
                    columns_to_keep = [line.strip() for line in f if line.strip()]
                if self.target_column not in columns_to_keep:
                    columns_to_keep.append(self.target_column)
                self.df = self.df[columns_to_keep]
            except FileNotFoundError:
                print(f"File not found: {self.file_path}")
                return None
            except Exception as e:
                print(f"An error occurred while reading the file: {e}")
                return None

        # List the column names
        columns = self.df.columns.tolist()
        print("Column names:")
        for col in columns:
            print(col)
        return columns

    def inverse_transform_y(self, y_scaled):
        # Inverse transform the scaled target variable
        return self.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()

if __name__ == "__main__":
    # Example usage of DataProcessor when run independently
    file_path = 'data/Aframax/P data_20200213-20200726_Democritos.csv'  # Update with your actual file path
    target_column = 'Power'  # The target column is always 'Power'
    keep_columns_file = 'columns_to_keep.txt'  # Path to the text file with columns to keep

    # Initialize DataProcessor with the new parameter
    data_processor = DataProcessor(
        file_path,
        target_column,
        keep_columns_file=keep_columns_file,
        fill_missing_with_median=True,   # Set to True to fill missing values with median
        exclude_missing_Hs=True          # Set to True to exclude rows with missing H_s
    )

    # Load and prepare data
    result = data_processor.load_and_prepare_data()
    if result is not None:
        X_train, X_test, X_train_unscaled, X_test_unscaled, y_train, y_test, y_train_unscaled, y_test_unscaled = result

        # Print dataset shapes and heads
        data_processor.print_dataset_shapes(X_train, X_test)
        # data_processor.print_dataset_head(X_train, X_test)

        # List all column names
        data_processor.list_column_names()