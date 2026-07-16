"""
Data Loading and Preprocessing Module

This module handles loading and preprocessing of both datasets:
1. Adult Income Dataset (benchmark)
2. AI Fair Recruitment Dataset (real-world)

Features:
- Missing value handling
- Categorical variable encoding (OneHotEncoding)
- Numerical feature normalization
- Protected attribute identification
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Handles data loading and preprocessing for bias detection in hiring systems.
    """
    
    def __init__(self):
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.scaler = StandardScaler()
        self.protected_attributes = []
        self.feature_columns = []
        self.target_column = ""
        
    def load_adult_income_dataset(self, url: str = None) -> pd.DataFrame:
        """
        Load the Adult Income Dataset.
        
        Args:
            url: Optional URL to fetch data from
            
        Returns:
            Loaded DataFrame
        """
        if url is None:
            # Use standard UCI repository URL
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
            
        column_names = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
        ]
        
        try:
            df = pd.read_csv(url, names=column_names, na_values=' ?', skipinitialspace=True)
            print(f"Adult Income Dataset loaded: {df.shape}")
            return df
        except Exception as e:
            raise FileNotFoundError(
                "Real dataset required. Failed to load Adult Income dataset. "
                "Please provide a valid dataset URL or local path."
            ) from e
    
    def load_ai_fair_dataset(self, file_path: str = None) -> pd.DataFrame:
        """
        Load the AI Fair Recruitment Dataset.
        
        Args:
            file_path: Path to the dataset file
            
        Returns:
            Loaded DataFrame
        """
        if file_path is None:
            raise FileNotFoundError(
                "Real dataset required. Please provide AI Fair Recruitment dataset path."
            )
        
        try:
            df = pd.read_csv(file_path)
            print(f"AI Fair Recruitment Dataset loaded: {df.shape}")
            return df
        except Exception as e:
            raise FileNotFoundError(
                f"AI Fair Recruitment dataset not found at: {file_path}"
            ) from e
    
    def identify_protected_attributes(self, df: pd.DataFrame, dataset_type: str) -> List[str]:
        """
        Identify protected attributes based on dataset type.
        
        Args:
            df: Input DataFrame
            dataset_type: 'adult' or 'ai_fair'
            
        Returns:
            List of protected attribute column names
        """
        if dataset_type == 'adult':
            protected = ['sex', 'race']
        elif dataset_type == 'ai_fair':
            protected = ['gender', 'ethnicity']
        else:
            # Auto-detect common protected attributes
            common_protected = ['sex', 'gender', 'race', 'ethnicity', 'age']
            protected = [col for col in common_protected if col in df.columns]
        
        self.protected_attributes = protected
        print(f"Identified protected attributes: {protected}")
        return protected
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with handled missing values
        """
        print(f"Missing values before handling:\n{df.isnull().sum()}")
        
        # Separate numerical and categorical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Fill numerical missing values with median
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
        
        # Fill categorical missing values with mode
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
        
        print(f"Missing values after handling:\n{df.isnull().sum()}")
        return df
    
    def encode_categorical_variables(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical variables using OneHotEncoding.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit the encoder (True for training, False for testing)
            
        Returns:
            DataFrame with encoded categorical variables
        """
        # Identify categorical columns (excluding target)
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target column if it's categorical
        if self.target_column in categorical_cols:
            categorical_cols.remove(self.target_column)
        
        if not categorical_cols:
            return df
        
        print(f"Encoding categorical columns: {categorical_cols}")
        
        if fit:
            # Fit and transform for training data
            encoded_data = self.encoder.fit_transform(df[categorical_cols])
            self.feature_columns = categorical_cols
        else:
            # Only transform for test data
            encoded_data = self.encoder.transform(df[categorical_cols])
        
        # Create feature names
        feature_names = []
        for i, col in enumerate(categorical_cols):
            categories = self.encoder.categories_[i]
            feature_names.extend([f"{col}_{cat}" for cat in categories])
        
        # Create DataFrame with encoded data
        encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=df.index)
        
        # Drop original categorical columns and concatenate encoded data
        df_encoded = df.drop(columns=categorical_cols)
        df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
        
        print(f"Shape after encoding: {df_encoded.shape}")
        return df_encoded
    
    def normalize_numerical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Normalize numerical features using StandardScaler.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit the scaler (True for training, False for testing)
            
        Returns:
            DataFrame with normalized numerical features
        """
        # Identify numerical columns (excluding target)
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target column if it's numerical
        if self.target_column in numerical_cols:
            numerical_cols.remove(self.target_column)
        
        # Remove ID columns and other non-feature numerical columns
        exclude_cols = ['applicant_id', 'fnlwgt']
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        if not numerical_cols:
            return df
        
        print(f"Normalizing numerical columns: {numerical_cols}")
        
        if fit:
            # Fit and transform for training data
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        else:
            # Only transform for test data
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        return df
    
    def prepare_target_variable(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Prepare target variable for binary classification.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            
        Returns:
            DataFrame with binary target variable
        """
        self.target_column = target_col
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")
        
        # Handle different target formats
        if df[target_col].dtype == 'object':
            # For categorical targets (like income)
            unique_values = df[target_col].unique()
            print(f"Unique target values: {unique_values}")
            
            # Convert to binary (1 for positive class, 0 for negative)
            if target_col == 'income':
                df[target_col] = (df[target_col] == '>50K').astype(int)
            else:
                # Generic binary conversion
                df[target_col] = (df[target_col] == unique_values[-1]).astype(int)
        
        print(f"Target distribution:\n{df[target_col].value_counts(normalize=True)}")
        return df
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets.
        
        Args:
            df: Input DataFrame
            test_size: Proportion of test data
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, test_df)
        """
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=df[self.target_column]
        )
        
        print(f"Training set shape: {train_df.shape}")
        print(f"Test set shape: {test_df.shape}")
        
        return train_df, test_df
    
    def preprocess_adult_income(self, url: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Complete preprocessing pipeline for Adult Income Dataset.
        
        Args:
            url: Optional URL for dataset
            
        Returns:
            Tuple of (train_df, test_df)
        """
        print("=" * 50)
        print("PREPROCESSING ADULT INCOME DATASET")
        print("=" * 50)
        
        # Load data
        df = self.load_adult_income_dataset(url)
        
        # Identify protected attributes
        self.identify_protected_attributes(df, 'adult')
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Prepare target variable
        df = self.prepare_target_variable(df, 'income')
        
        # Split data
        train_df, test_df = self.split_data(df)
        
        # Encode categorical variables
        train_df = self.encode_categorical_variables(train_df, fit=True)
        test_df = self.encode_categorical_variables(test_df, fit=False)
        
        # Normalize numerical features
        train_df = self.normalize_numerical_features(train_df, fit=True)
        test_df = self.normalize_numerical_features(test_df, fit=False)
        
        print("Adult Income preprocessing completed!")
        return train_df, test_df
    
    def preprocess_ai_fair(self, file_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Complete preprocessing pipeline for AI Fair Recruitment Dataset.
        
        Args:
            file_path: Optional path for dataset
            
        Returns:
            Tuple of (train_df, test_df)
        """
        print("=" * 50)
        print("PREPROCESSING AI FAIR RECRUITMENT DATASET")
        print("=" * 50)
        
        # Load data
        df = self.load_ai_fair_dataset(file_path)
        
        # Identify protected attributes
        self.identify_protected_attributes(df, 'ai_fair')
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Prepare target variable
        df = self.prepare_target_variable(df, 'hired')
        
        # Drop ID column
        if 'applicant_id' in df.columns:
            df = df.drop(columns=['applicant_id'])
        
        # Split data
        train_df, test_df = self.split_data(df)
        
        # Encode categorical variables
        train_df = self.encode_categorical_variables(train_df, fit=True)
        test_df = self.encode_categorical_variables(test_df, fit=False)
        
        # Normalize numerical features
        train_df = self.normalize_numerical_features(train_df, fit=True)
        test_df = self.normalize_numerical_features(test_df, fit=False)
        
        print("AI Fair Recruitment preprocessing completed!")
        return train_df, test_df


def main():
    """
    Test the preprocessing module.
    """
    preprocessor = DataPreprocessor()
    
    # Test Adult Income preprocessing
    adult_train, adult_test = preprocessor.preprocess_adult_income()
    print(f"Adult train shape: {adult_train.shape}")
    print(f"Adult test shape: {adult_test.shape}")
    
    # Test AI Fair preprocessing
    ai_train, ai_test = preprocessor.preprocess_ai_fair()
    print(f"AI Fair train shape: {ai_train.shape}")
    print(f"AI Fair test shape: {ai_test.shape}")


if __name__ == "__main__":
    main()
