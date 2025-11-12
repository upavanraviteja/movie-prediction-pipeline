"""
Data preprocessing module for movie box office prediction pipeline.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, List
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Data preprocessing class for movie box office prediction.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the preprocessor with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        if config_path:
            with open(config_path, 'r') as file:
                self.config = json.load(file)
        else:
            self.config = {}
        
        self.log_transform_columns = self.config.get('feature_engineering', {}).get(
            'log_transform_columns', ['box', 'budget', 'starpowr', 'addict', 'cmngsoon', 'fandango', 'cntwait3']
        )
        self.skewness_threshold = self.config.get('feature_engineering', {}).get(
            'skewness_threshold', 1.0
        )
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame with loaded data
        """
        try:
            logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def analyze_skewness(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Analyze skewness and kurtosis of specified columns.
        
        Args:
            df: Input DataFrame
            columns: List of columns to analyze
            
        Returns:
            DataFrame with skewness and kurtosis statistics
        """
        logger.info("Analyzing skewness and kurtosis")
        stats = df[columns].agg(['skew', 'kurtosis']).transpose()
        logger.info(f"Skewness analysis completed:\n{stats}")
        return stats
    
    def apply_log_transformation(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Apply log transformation to specified columns.
        
        Args:
            df: Input DataFrame
            columns: List of columns to transform
            
        Returns:
            DataFrame with log-transformed columns
        """
        df_transformed = df.copy()
        
        for col in columns:
            if col in df_transformed.columns:
                # Check if column has positive values
                if (df_transformed[col] > 0).all():
                    new_col_name = f'log_{col}'
                    df_transformed[new_col_name] = np.log(df_transformed[col])
                    logger.info(f"Applied log transformation to {col} -> {new_col_name}")
                else:
                    logger.warning(f"Column {col} contains non-positive values. Skipping log transformation.")
        
        return df_transformed
    
    def identify_skewed_columns(self, df: pd.DataFrame, columns: List[str]) -> List[str]:
        """
        Identify columns that are highly skewed based on threshold.
        
        Args:
            df: Input DataFrame
            columns: List of columns to check
            
        Returns:
            List of highly skewed columns
        """
        skewed_cols = []
        for col in columns:
            if col in df.columns:
                skewness = df[col].skew()
                if abs(skewness) > self.skewness_threshold:
                    skewed_cols.append(col)
                    logger.info(f"Column {col} is highly skewed (skewness: {skewness:.3f})")
        
        return skewed_cols
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the data by handling missing values and outliers.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Handle missing values
        initial_shape = df_clean.shape
        df_clean = df_clean.dropna()
        final_shape = df_clean.shape
        
        if initial_shape != final_shape:
            logger.info(f"Removed {initial_shape[0] - final_shape[0]} rows with missing values")
        
        return df_clean
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Starting data preprocessing pipeline")
        
        # Clean data
        df_processed = self.clean_data(df)
        
        # Analyze skewness for continuous variables
        if all(col in df_processed.columns for col in self.log_transform_columns):
            self.analyze_skewness(df_processed, self.log_transform_columns)
        
        # Apply log transformation to skewed variables
        skewed_cols = self.identify_skewed_columns(df_processed, self.log_transform_columns)
        if skewed_cols:
            df_processed = self.apply_log_transformation(df_processed, skewed_cols)
        
        logger.info("Data preprocessing completed")
        return df_processed


def main():
    """
    Main function for standalone execution.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Data Preprocessing')
    parser.add_argument('--input-path', type=str, required=True, help='Input data path')
    parser.add_argument('--output-path', type=str, required=True, help='Output data path')
    parser.add_argument('--config-path', type=str, help='Configuration file path')
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(args.config_path)
    
    # Load and preprocess data
    df = preprocessor.load_data(args.input_path)
    df_processed = preprocessor.preprocess(df)
    
    # Save processed data
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    df_processed.to_csv(args.output_path, index=False)
    logger.info(f"Processed data saved to {args.output_path}")


if __name__ == "__main__":
    main()
