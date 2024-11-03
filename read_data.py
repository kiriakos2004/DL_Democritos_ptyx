import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    def __init__(
        self, file_path, target_column, drop_columns=None, test_size=0.2, random_state=42
    ):
        self.file_path = file_path
        self.target_column = target_column
        self.drop_columns = drop_columns
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
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

        # Drop specified columns, if provided
        if self.drop_columns is not None:
            for col in self.drop_columns:
                if col in self.df.columns:
                    self.df.drop(columns=col, inplace=True)
                else:
                    print(f"Warning: Column '{col}' not found in the dataset.")

        # Ensure the target column exists
        if self.target_column not in self.df.columns:
            print(f"Error: Target column '{self.target_column}' not found in the dataset.")
            return None

        # Drop rows where 'Power' is less than 1000
        self.df = self.df[self.df['Power'] >= 1000]

        # Drop rows where 'Speed-Through-Water' is less than 4 knots
        self.df = self.df[self.df['Speed-Through-Water'] >= 4]

        # Check for missing values and fill them
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())

        # Split the data into features (X) and target (y)
        X = self.df.drop(self.target_column, axis=1)  # Features
        y = self.df[self.target_column]  # Target

        # Keep a copy of unscaled features for physics-based loss calculations
        X_unscaled = X.copy()

        # Split the dataset into training and testing sets without shuffling
        X_train_unscaled, X_test_unscaled, y_train, y_test = train_test_split(
            X_unscaled, y, test_size=self.test_size, random_state=self.random_state, shuffle=False
        )

        # Scale the features using the training data
        self.scaler.fit(X_train_unscaled)
        X_train_scaled = self.scaler.transform(X_train_unscaled)
        X_test_scaled = self.scaler.transform(X_test_unscaled)

        # Convert scaled features back to DataFrames to maintain column names
        X_train = pd.DataFrame(X_train_scaled, columns=X_unscaled.columns)
        X_test = pd.DataFrame(X_test_scaled, columns=X_unscaled.columns)

        return X_train, X_test, X_train_unscaled, X_test_unscaled, y_train, y_test

    def print_dataset_shapes(self, X_train, X_test):
        # Print the shape of the datasets
        print(f"Training features shape: {X_train.shape}")
        print(f"Test features shape: {X_test.shape}")

    def print_dataset_head(self, X_train, X_test):
        # Print the first few rows of X_train and X_test
        print("First few rows of training features (X_train):")
        print(X_train.head())  # X_train is a DataFrame now

        print("\nFirst few rows of test features (X_test):")
        print(X_test.head())  # X_test is a DataFrame now

    def list_column_names(self):
        # Ensure the DataFrame is loaded
        if self.df is None:
            try:
                self.df = pd.read_csv(self.file_path)
                if self.drop_columns is not None:
                    self.df.drop(columns=self.drop_columns, inplace=True)
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
        return columns  # Optional: return the list of column names

if __name__ == "__main__":
    # Example usage of DataProcessor when run independently
    file_path = 'data/Aframax/P data_20200213-20200726_Democritos.csv'  # Update with your actual file path
    target_column = 'Power'  # Update with your actual target column
    drop_columns = ['TIME']  # Update with columns you wish to drop

    # Initialize DataProcessor
    data_processor = DataProcessor(file_path, target_column, drop_columns)

    # Load and prepare data
    result = data_processor.load_and_prepare_data()
    if result is not None:
        X_train, X_test, X_train_unscaled, X_test_unscaled, y_train, y_test = result

        # Print dataset shapes and heads
        data_processor.print_dataset_shapes(X_train, X_test)
        #data_processor.print_dataset_head(X_train, X_test)

        # List all column names
        # data_processor.list_column_names()