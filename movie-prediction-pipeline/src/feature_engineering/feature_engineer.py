"""
Feature engineering module for movie box office prediction pipeline.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, List, Dict
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering class for movie box office prediction.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the feature engineer with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        if config_path:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        else:
            self.config = {}
        
        self.pca_components = self.config.get('feature_engineering', {}).get('pca_components', 3)
        self.variance_threshold = self.config.get('feature_engineering', {}).get('variance_threshold', 0.8)
        self.target_column = self.config.get('model', {}).get('target_column', 'log_box')
        self.drop_columns = self.config.get('model', {}).get('drop_columns', [])
        
        self.scaler = None
        self.pca = None
    
    def prepare_features_and_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target variable.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        logger.info("Preparing features and target variable")
        
        # Create target variable
        if self.target_column not in df.columns:
            logger.error(f"Target column {self.target_column} not found in DataFrame")
            raise ValueError(f"Target column {self.target_column} not found")
        
        y = df[self.target_column].copy()
        
        # Prepare features
        X = df.copy()
        
        # Drop target column and specified columns
        columns_to_drop = [self.target_column] + self.drop_columns
        columns_to_drop = [col for col in columns_to_drop if col in X.columns]
        X = X.drop(columns=columns_to_drop)
        
        # Add constant term for regression
        X['const'] = 1.0
        
        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        logger.info(f"Feature columns: {list(X.columns)}")
        
        return X, y


    def apply_pca(self, X: pd.DataFrame, fit: bool = True, n_components: int = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply Principal Component Analysis to features.
        
        Args:
            X: Input features DataFrame
            fit: Whether to fit the PCA or use existing fitted PCA
            n_components: Number of components to use (if None, use all features)
            
        Returns:
            Tuple of (PCA-transformed DataFrame, PCA info dictionary)
        """
        logger.info("Applying PCA transformation")
        
        if fit:
            # Standardize the data
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Apply sample standard deviation correction
            sample_std = np.std(X_scaled, axis=0, ddof=1)
            X_scaled = X_scaled / sample_std
            
            # Fit PCA - use all components if n_components not specified
            pca_components = n_components if n_components is not None else min(X.shape[0], X.shape[1])
            self.pca = PCA(n_components=pca_components)
            X_pca = self.pca.fit_transform(X_scaled)
            
        else:
            if self.scaler is None or self.pca is None:
                raise ValueError("PCA and scaler must be fitted before transform")
            
            X_scaled = self.scaler.transform(X)
            sample_std = np.std(X_scaled, axis=0, ddof=1)
            X_scaled = X_scaled / sample_std
            X_pca = self.pca.transform(X_scaled)
        
        # Create DataFrame with PCA components
        pc_columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
        X_pca_df = pd.DataFrame(X_pca, columns=pc_columns, index=X.index)
        
        # Calculate PCA information
        pca_info = {
            'explained_variance_ratio': self.pca.explained_variance_ratio_,
            'explained_variance': self.pca.explained_variance_,
            'cumulative_variance_ratio': np.cumsum(self.pca.explained_variance_ratio_),
            'components': self.pca.components_,
            'n_components': self.pca.n_components_
        }
        
        logger.info(f"PCA applied. Components: {X_pca.shape[1]}")
        logger.info(f"Explained variance ratio: {pca_info['explained_variance_ratio']}")
        logger.info(f"Cumulative variance ratio: {pca_info['cumulative_variance_ratio']}")
        
        return X_pca_df, pca_info
    
    def select_pca_components(self, pca_info: Dict, method: str = 'variance_threshold') -> int:
        """
        Select optimal number of PCA components based on different criteria.
        
        Args:
            pca_info: PCA information dictionary
            method: Selection method ('variance_threshold', 'kaiser', 'elbow')
            
        Returns:
            Number of components to select
        """
        if method == 'variance_threshold':
            # Select components that explain specified variance threshold
            cumulative_var = pca_info['cumulative_variance_ratio']
            n_components = np.argmax(cumulative_var >= self.variance_threshold) + 1
            
        elif method == 'kaiser':
            # Kaiser's rule: select components with eigenvalue > 1
            eigenvalues = pca_info['explained_variance']
            n_components = np.sum(eigenvalues > 1)
            
        else:
            # Default to configured number of components
            n_components = self.pca_components
        
        logger.info(f"Selected {n_components} components using {method} method")
        return n_components
    
    def create_combined_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create combined features with PCA applied to all features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (PCA-transformed features DataFrame, target Series)
        """
        logger.info("Creating combined features with PCA applied to all features")
        
        # Get all features and target
        X, y = self.prepare_features_and_target(df)
        
        if X.empty:
            logger.warning("No features available for PCA")
            return pd.DataFrame(), pd.Series()
        
        # Apply PCA to all features (no component limit initially)
        features_pca_full, pca_info = self.apply_pca(X, fit=True, n_components=None)
        
        # Select optimal number of PCA components using variance threshold
        optimal_components = self.select_pca_components(pca_info, method='variance_threshold')
        
        # Select only the optimal number of components from the full PCA result
        features_pca_selected = features_pca_full.iloc[:, :optimal_components]
        
        logger.info(f"PCA features shape: {features_pca_selected.shape}")
        logger.info(f"Used {optimal_components} PCA components out of {features_pca_full.shape[1]} total")
        return features_pca_selected, y
    
    def save_feature_artifacts(self, output_dir: str):
        """
        Save feature engineering artifacts (scaler, PCA).
        
        Args:
            output_dir: Directory to save artifacts
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if self.scaler is not None:
            scaler_path = os.path.join(output_dir, 'scaler.joblib')
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Scaler saved to {scaler_path}")
        
        if self.pca is not None:
            pca_path = os.path.join(output_dir, 'pca.joblib')
            joblib.dump(self.pca, pca_path)
            logger.info(f"PCA saved to {pca_path}")
    
    def load_feature_artifacts(self, input_dir: str):
        """
        Load feature engineering artifacts (scaler, PCA).
        
        Args:
            input_dir: Directory to load artifacts from
        """
        scaler_path = os.path.join(input_dir, 'scaler.joblib')
        pca_path = os.path.join(input_dir, 'pca.joblib')
        
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Scaler loaded from {scaler_path}")
        
        if os.path.exists(pca_path):
            self.pca = joblib.load(pca_path)
            logger.info(f"PCA loaded from {pca_path}")


def main():
    """
    Main function for standalone execution.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Feature Engineering')
    parser.add_argument('--input-path', type=str, required=True, help='Input data path')
    parser.add_argument('--output-path', type=str, help='Output path for train/test data (legacy)')
    parser.add_argument('--train-features-path', type=str, help='Train features output path')
    parser.add_argument('--train-target-path', type=str, help='Train target output path')
    parser.add_argument('--test-features-path', type=str, help='Test features output path')
    parser.add_argument('--test-target-path', type=str, help='Test target output path')
    parser.add_argument('--artifacts-path', type=str, required=True, help='Artifacts output path')
    parser.add_argument('--config-path', type=str, help='Configuration file path')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size (default: 0.2)')
    parser.add_argument('--random-state', type=int, default=42, help='Random state for reproducibility')
    
    args = parser.parse_args()
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer(args.config_path)
    
    # Load data
    df = pd.read_csv(args.input_path)
    logger.info(f"Loaded data with shape: {df.shape}")
    
    # Create combined features
    X_combined, y = feature_engineer.create_combined_features(df)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, 
        test_size=args.test_size, 
        random_state=args.random_state,
        stratify=None  # Can't stratify continuous target
    )
    
    logger.info(f"Train set shape: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Test set shape: X={X_test.shape}, y={y_test.shape}")
    
    # Determine output paths
    if args.train_features_path and args.train_target_path and args.test_features_path and args.test_target_path:
        # Use explicit paths (SageMaker mode)
        train_features_path = args.train_features_path
        train_target_path = args.train_target_path
        test_features_path = args.test_features_path
        test_target_path = args.test_target_path
    elif args.output_path:
        # Use legacy output path (local mode)
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        train_features_path = args.output_path.replace('.csv', '_train_features.csv')
        train_target_path = args.output_path.replace('.csv', '_train_target.csv')
        test_features_path = args.output_path.replace('.csv', '_test_features.csv')
        test_target_path = args.output_path.replace('.csv', '_test_target.csv')
    else:
        raise ValueError("Either --output-path or all of --train-features-path, --train-target-path, --test-features-path, --test-target-path must be provided")
    
    # Create output directories
    for path in [train_features_path, train_target_path, test_features_path, test_target_path]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save train and test sets
    X_train.to_csv(train_features_path, index=False)
    y_train.to_csv(train_target_path, index=False)
    X_test.to_csv(test_features_path, index=False)
    y_test.to_csv(test_target_path, index=False)
    
    # Save feature engineering artifacts
    feature_engineer.save_feature_artifacts(args.artifacts_path)
    
    logger.info(f"Train features saved to {train_features_path}")
    logger.info(f"Train target saved to {train_target_path}")
    logger.info(f"Test features saved to {test_features_path}")
    logger.info(f"Test target saved to {test_target_path}")
    logger.info(f"Artifacts saved to {args.artifacts_path}")


if __name__ == "__main__":
    main()
