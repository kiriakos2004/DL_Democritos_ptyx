import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from math import ceil

class DataProcessor:
    def __init__(
        self, file_path, target_column, keep_columns_file, test_size=0.1, random_state=42,
        fill_missing_with_median=True, exclude_missing_Hs=True, train_fraction=1.0, test_segment=9
    ):
        self.file_path = file_path
        self.target_column = target_column
        self.keep_columns_file = keep_columns_file
        self.test_size = test_size
        self.random_state = random_state
        self.fill_missing_with_median = fill_missing_with_median
        self.exclude_missing_Hs = exclude_missing_Hs
        self.train_fraction = train_fraction
        self.test_segment = test_segment
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.df = None

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

        # Check for missing values and fill them
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns

        # Fill missing values with median or mean
        if self.fill_missing_with_median:
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        else:
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())

        # Calculate the size of each segment
        total_rows = len(self.df)
        segment_size = ceil(total_rows / 10)

        # Determine start and end indices for the selected test segment
        start_idx = self.test_segment * segment_size
        end_idx = min(start_idx + segment_size, total_rows)

        # Split the data into testing and training sets
        test_indices = range(start_idx, end_idx)
        train_indices = [i for i in range(total_rows) if i not in test_indices]

        X = self.df.drop(self.target_column, axis=1)  # Features
        y = self.df[[self.target_column]]  # Target (as DataFrame for scaling)

        X_train_unscaled = X.iloc[train_indices]
        X_test_unscaled = X.iloc[test_indices]
        y_train_unscaled = y.iloc[train_indices]
        y_test_unscaled = y.iloc[test_indices]

        # Further split the training set based on train_fraction
        if self.train_fraction < 1.0:
            train_fraction_size = self.train_fraction  # Fraction of training data to use
            X_train_unscaled, X_unused_unscaled, y_train_unscaled, y_unused_unscaled = train_test_split(
                X_train_unscaled, y_train_unscaled, 
                test_size=(1 - train_fraction_size), 
                random_state=self.random_state, 
                shuffle=True
    )

        # Scale the features using the actual training data
        self.scaler_X.fit(X_train_unscaled)
        X_train_scaled = self.scaler_X.transform(X_train_unscaled)
        X_test_scaled = self.scaler_X.transform(X_test_unscaled)

        # Scale the target variable using the actual training data
        self.scaler_y.fit(y_train_unscaled)
        y_train_scaled = self.scaler_y.transform(y_train_unscaled)
        y_test_scaled = self.scaler_y.transform(y_test_unscaled)

        # Convert scaled features back to DataFrames to maintain column names
        X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_test = pd.DataFrame(X_test_scaled, columns=X.columns)

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
        print(f"Training features shape: {X_train.shape}")
        print(f"Test features shape: {X_test.shape}")


    #def print_dataset_head(self, X_train, X_test):
    #    print("First few rows of training features (X_train):")
    #    print(X_train.head())
    #    print("\nFirst few rows of test features (X_test):")
    #    print(X_test.head())


    def list_column_names(self):
        if self.df is None:
            try:
                self.df = pd.read_csv(self.file_path)
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

        columns = self.df.columns.tolist()
        print("Column names:")
        for col in columns:
            print(col)
        return columns

    def inverse_transform_y(self, y_scaled):
        return self.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()


if __name__ == "__main__":
    file_path = 'data/Aframax/P data_20200213-20200726_Democritos.csv'
    target_column = 'Power'
    keep_columns_file = 'columns_to_keep.txt'

    data_processor = DataProcessor(
        file_path,
        target_column,
        keep_columns_file=keep_columns_file,
        fill_missing_with_median=True,
        exclude_missing_Hs=True,
        train_fraction=0.5,
        test_segment=9
    )

    result = data_processor.load_and_prepare_data()
    if result:
        X_train, X_test, X_train_unscaled, X_test_unscaled, y_train, y_test, y_train_unscaled, y_test_unscaled = result
        data_processor.print_dataset_shapes(X_train, X_test)
        data_processor.list_column_names()