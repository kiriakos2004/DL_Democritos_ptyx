import pandas as pd
from sklearn.preprocessing import StandardScaler
from math import ceil

class DataProcessor:
    def __init__(
        self, 
        file_path, 
        target_column, 
        keep_columns_file,
        fill_missing_with_median=True, 
        exclude_missing_Hs=True
    ):
        self.file_path = file_path
        self.target_column = target_column
        self.keep_columns_file = keep_columns_file
        self.fill_missing_with_median = fill_missing_with_median
        self.exclude_missing_Hs = exclude_missing_Hs

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

        # -----------------------------------
        # 4) Fill missing numeric values
        # -----------------------------------
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        if self.fill_missing_with_median:
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        else:
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())

        # -----------------------------------
        # 5) Manual train/test split by row index
        #    Rows 0–9,999  -> training
        #    Rows 10,100–10,999 -> testing
        # -----------------------------------
        total_rows = len(self.df)
        if total_rows < 11000:
            print(f"Warning: Your dataset has only {total_rows} rows. "
                  f"Cannot reliably split at [0:10000] and [10100:11000].")
            return None

        train_indices = range(0, 10000)
        test_indices = range(10010, 11000)

        X = self.df.drop(self.target_column, axis=1)
        y = self.df[[self.target_column]]  # keep y as DataFrame for scaling

        # Unscaled splits
        X_train_unscaled = X.iloc[train_indices]
        X_test_unscaled = X.iloc[test_indices]
        y_train_unscaled = y.iloc[train_indices]
        y_test_unscaled = y.iloc[test_indices]

        # -----------------------------------
        # 6) Scale X using training data
        # -----------------------------------
        self.scaler_X.fit(X_train_unscaled)
        X_train_scaled = self.scaler_X.transform(X_train_unscaled)
        X_test_scaled = self.scaler_X.transform(X_test_unscaled)

        # -----------------------------------
        # 7) Scale y using training data
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
        """
        Prints the shapes of the training and test feature sets.
        """
        print(f"Training features shape: {X_train.shape}")
        print(f"Test features shape: {X_test.shape}")

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
        """
        Inverse-transforms scaled target data back to the original scale.
        """
        return self.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()


if __name__ == "__main__":
    file_path = 'data/Dan/P data_20210428-20211111_Democritos.csv'
    target_column = 'Power'
    keep_columns_file = 'columns_to_keep.txt'

    data_processor = DataProcessor(
        file_path=file_path,
        target_column=target_column,
        keep_columns_file=keep_columns_file,
        fill_missing_with_median=True,
        exclude_missing_Hs=True
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

    else:
        print("Error in loading and preparing data.")