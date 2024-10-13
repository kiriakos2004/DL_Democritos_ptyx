import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    def __init__(self, file_path, target_column, drop_columns=None, test_size=0.2, random_state=42):
        self.file_path = file_path
        self.target_column = target_column
        self.drop_columns = drop_columns
        self.test_size = test_size
        self.random_state = random_state

    def load_and_prepare_data(self):
        # Load CSV file into a DataFrame
        df = pd.read_csv(self.file_path)

        # Drop specified columns, if provided
        if self.drop_columns is not None:
            df = df.drop(columns=self.drop_columns)

        # Check for missing values and fill them
        missing_values_before = df.isnull().sum().sum()
        if missing_values_before > 0:
            df.fillna(df.mean(), inplace=True)
            missing_values_after = df.isnull().sum().sum()
            if missing_values_after > 0:
                print(f"There are still {missing_values_after} missing values remaining. Please check the dataset.")
        
        # Check for non-numeric columns
        non_numeric_columns = df.select_dtypes(exclude=['int64', 'float64']).columns
        if len(non_numeric_columns) > 0:
            print(f"Warning: Dataset contains non-numeric columns: {non_numeric_columns}. Please handle these manually.")
        
        # Split the data into features (X) and target (y)
        X = df.drop(self.target_column, axis=1)  # Features
        y = df[self.target_column]  # Target

        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=self.test_size, random_state=self.random_state)

        return X_train, X_test, y_train, y_test

    def print_dataset_shapes(self, X_train, X_test):
        # Print the shape of the datasets
        print(f"Training features shape: {X_train.shape}")
        print(f"Test features shape: {X_test.shape}")

    def print_dataset_head(self, X_train, X_test):
        # Print the first few rows of X_train and X_test
        print("First few rows of training features (X_train):")
        print(pd.DataFrame(X_train).head())  # Convert numpy array to DataFrame for better readability

        print("\nFirst few rows of test features (X_test):")
        print(pd.DataFrame(X_test).head())  # Convert numpy array to DataFrame for better readability

if __name__ == "__main__":
    # Example usage of DataProcessor when run independently
    file_path = 'data/Dan/P data_20210428-20211111_Democritos.csv'
    target_column = 'Speed-Through-Water'
    drop_columns = ['TIME']

    # Initialize DataProcessor
    data_processor = DataProcessor(file_path, target_column, drop_columns)
    
    # Load and prepare data
    X_train, X_test, y_train, y_test = data_processor.load_and_prepare_data()

    # Print dataset shapes and heads
    data_processor.print_dataset_shapes(X_train, X_test)
    data_processor.print_dataset_head(X_train, X_test)
