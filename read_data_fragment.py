import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from math import ceil

class DataProcessor:
    def __init__(
        self, 
        file_path, 
        target_column, 
        keep_columns_file, 
        test_size=0.1, 
        random_state=42,
        fill_missing_with_median=True, 
        exclude_missing_Hs=True, 
        train_fraction=1.0, 
        test_segment=5
    ):
        """
        file_path: Path to the CSV dataset.
        target_column: Name of the column to predict (e.g., 'Power').
        keep_columns_file: Path to a text file listing columns to keep.
        test_size: Fraction/percentage for splitting data (but here you use a segment approach, so may not apply).
        random_state: Random seed for reproducibility.
        fill_missing_with_median: If True, fill numeric NaNs with median, else mean.
        exclude_missing_Hs: If True, drop rows missing 'Significant_Wave_Height'.
        train_fraction: Fraction of the training data to keep (if < 1, it further splits training).
        test_segment: Which segment (0–9) to pick for testing if you’re doing a 10-segment approach.
        """
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
        
        self.df = None  # Will store the filtered DataFrame

    def load_and_prepare_data(self):
        # -----------------------------------
        # 1) Load CSV file into DataFrame
        # -----------------------------------
        try:
            self.df = pd.read_csv(self.file_path)
        except FileNotFoundError:
            print(f"File not found: {self.file_path}")
            return None
        except Exception as e:
            print(f"An error occurred while reading the file: {e}")
            return None

        # -----------------------------------
        # 2) Read columns to keep
        # -----------------------------------
        try:
            with open(self.keep_columns_file, 'r') as f:
                columns_to_keep = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"Columns file not found: {self.keep_columns_file}")
            return None
        except Exception as e:
            print(f"An error occurred while reading the columns file: {e}")
            return None

        # Ensure target_column is included
        if self.target_column not in columns_to_keep:
            columns_to_keep.append(self.target_column)

        # Check if columns exist in the DataFrame
        missing_columns = [col for col in columns_to_keep if col not in self.df.columns]
        if missing_columns:
            print(f"Error: The following columns are not in the dataset: {missing_columns}")
            return None

        # Filter DataFrame to keep specified columns
        self.df = self.df[columns_to_keep]

        # Ensure target column exists
        if self.target_column not in self.df.columns:
            print(f"Error: Target column '{self.target_column}' not found in the dataset.")
            return None

        # -----------------------------------
        # 3) Optionally exclude rows with missing H_s
        # -----------------------------------
        if self.exclude_missing_Hs:
            if 'Significant_Wave_Height' in self.df.columns:
                initial_row_count = len(self.df)
                self.df = self.df.dropna(subset=['Significant_Wave_Height'])
                rows_dropped = initial_row_count - len(self.df)
                print(f"Dropped {rows_dropped} rows due to missing 'Significant_Wave_Height'")
            else:
                print("Warning: 'Significant_Wave_Height' column not found in the dataset.")
        '''
        # -----------------------------------
        # 4) Remove contradictory rows:
        #    e.g., if 'Power' == 0 but 'Speed-Through-Water' > 6.0
        # -----------------------------------
        if 'Speed-Through-Water' in self.df.columns:
            initial_count = len(self.df)
            self.df = self.df[~((self.df[self.target_column] == 0) & (self.df['Speed-Through-Water'] > 6.0))]
            removed_rows = initial_count - len(self.df)
            print(f"Removed {removed_rows} rows where Speed>6 knot but Power=0.")
        else:
            print("Warning: 'Speed-Through-Water' column not found in the dataset. No contradictory rows removed.")
        '''
        # -----------------------------------
        # 5) Fill missing numeric values
        # -----------------------------------
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        if self.fill_missing_with_median:
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        else:
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())

        # -----------------------------------
        # 6) Segment-based train/test split
        # -----------------------------------
        total_rows = len(self.df)
        segment_size = ceil(total_rows / 10)

        start_idx = self.test_segment * segment_size
        end_idx = min(start_idx + segment_size, total_rows)

        test_indices = range(start_idx, end_idx)
        train_indices = [i for i in range(total_rows) if i not in test_indices]

        X = self.df.drop(self.target_column, axis=1)
        y = self.df[[self.target_column]]  # DataFrame for scaling

        X_train_unscaled = X.iloc[train_indices]
        X_test_unscaled = X.iloc[test_indices]
        y_train_unscaled = y.iloc[train_indices]
        y_test_unscaled = y.iloc[test_indices]

        # -----------------------------------
        # 7) If train_fraction < 1, further split training
        # -----------------------------------
        if self.train_fraction < 1.0:
            X_train_unscaled, X_unused_unscaled, y_train_unscaled, y_unused_unscaled = train_test_split(
                X_train_unscaled, 
                y_train_unscaled, 
                test_size=(1 - self.train_fraction), 
                random_state=self.random_state, 
                shuffle=False
            )

        # -----------------------------------
        # 8) Scale X with training data
        # -----------------------------------
        self.scaler_X.fit(X_train_unscaled)
        X_train_scaled = self.scaler_X.transform(X_train_unscaled)
        X_test_scaled = self.scaler_X.transform(X_test_unscaled)

        # -----------------------------------
        # 9) Scale y with training data
        # -----------------------------------
        self.scaler_y.fit(y_train_unscaled)
        y_train_scaled = self.scaler_y.transform(y_train_unscaled)
        y_test_scaled = self.scaler_y.transform(y_test_unscaled)

        # Convert to DataFrame/Series for consistency
        X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_test = pd.DataFrame(X_test_scaled, columns=X.columns)

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

    def list_column_names(self):
        """
        Prints and returns the columns in self.df (or loads them if not set).
        """
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
        file_path=file_path,
        target_column=target_column,
        keep_columns_file=keep_columns_file,
        fill_missing_with_median=True,
        exclude_missing_Hs=True,
        train_fraction=1.0,
        test_segment=9
    )

    result = data_processor.load_and_prepare_data()
    if result:
        (
            X_train,
            X_test,
            X_train_unscaled,
            X_test_unscaled,
            y_train,
            y_test,
            y_train_unscaled,
            y_test_unscaled
        ) = result

        data_processor.print_dataset_shapes(X_train, X_test)
        data_processor.list_column_names()

        #print("\nSample of X_train_unscaled head:")
        #print(X_train_unscaled.head())

        #print("\nSample of y_train_unscaled head:")
        #print(y_train_unscaled.head())
    else:
        print("Error in loading and preparing data.")