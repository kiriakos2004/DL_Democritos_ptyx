import pandas as pd

class DataProcessor:
    def __init__(
        self, file_path, target_column, keep_columns_file, fill_missing_with_median=True, exclude_missing_Hs=True
    ):
        self.file_path = file_path
        self.target_column = target_column
        self.keep_columns_file = keep_columns_file  # Path to the text file with columns to keep
        self.fill_missing_with_median = fill_missing_with_median
        self.exclude_missing_Hs = exclude_missing_Hs
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

        # Proceed with data preparation
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
        initial_row_count = len(self.df)
        self.df = self.df[self.df['Power'] >= 1000]
        rows_dropped_power = initial_row_count - len(self.df)
        print(f"Dropped {rows_dropped_power} rows where 'Power' < 1000")

        # Drop rows where 'Speed-Through-Water' is less than 4 knots
        if 'Speed-Through-Water' in self.df.columns:
            initial_row_count = len(self.df)
            self.df = self.df[self.df['Speed-Through-Water'] >= 4]
            rows_dropped_speed = initial_row_count - len(self.df)
            print(f"Dropped {rows_dropped_speed} rows where 'Speed-Through-Water' < 4 knots")
        else:
            print("Warning: 'Speed-Through-Water' column not found in the dataset.")

        # Check for missing values and fill them
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns

        # Fill missing values with median or mean
        if self.fill_missing_with_median:
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        else:
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())

        return self.df

    def save_cleaned_data(self, output_file_path):
        # Ensure the data is loaded and prepared
        if self.df is None:
            self.load_and_prepare_data()
        # Save the cleaned DataFrame to CSV
        self.df.to_csv(output_file_path, index=False)
        print(f"Cleaned data saved to {output_file_path}")

if __name__ == "__main__":
    # Example usage of DataProcessor when run independently
    file_path = 'data/Aframax/P data_20200213-20200726_Democritos.csv'  # Update with your actual file path
    target_column = 'Power'  # The target column is always 'Power'
    keep_columns_file = 'columns_to_keep.txt'  # Path to the text file with columns to keep

    # Initialize DataProcessor with the specified parameters
    data_processor = DataProcessor(
        file_path,
        target_column,
        keep_columns_file=keep_columns_file,
        fill_missing_with_median=True,   # Set to True to fill missing values with median
        exclude_missing_Hs=True          # Set to True to exclude rows with missing H_s
    )

    # Load and prepare data
    result_df = data_processor.load_and_prepare_data()
    if result_df is not None:
        # Save the cleaned data to a new CSV file
        output_file_path = 'data/Aframax/P data_20200213-20200726_Democritos_cleaned.csv'  # Update with your desired output path
        data_processor.save_cleaned_data(output_file_path)