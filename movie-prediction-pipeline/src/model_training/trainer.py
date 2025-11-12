"""
Model training module for movie box office prediction pipeline.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Any
import json
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
import json
import subprocess
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Model training class for movie box office prediction.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the model trainer with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        if config_path:
            with open(config_path, 'r') as file:
                self.config = json.load(file)
        else:
            self.config = {}
        
        self.model_type = self.config.get('model', {}).get('type', 'linear_regression')
        self.significance_level = self.config.get('model', {}).get('significance_level', 0.1)
        self.train_test_split_ratio = self.config.get('data', {}).get('train_test_split', 0.8)
        self.random_state = self.config.get('data', {}).get('random_state', 42)
        
        self.model = None
        self.model_stats = None
        
    def train_statsmodels_ols(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, Dict]:
        """
        Train OLS regression model using statsmodels.
        
        Args:
            X: Features DataFrame
            y: Target Series
            
        Returns:
            Tuple of (fitted model, model statistics)
        """
        logger.info("Training OLS regression model using statsmodels")
        
        # Fit the model
        model = sm.OLS(y, X).fit()
        
        # Extract model statistics
        stats = {
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'f_statistic': model.fvalue,
            'f_pvalue': model.f_pvalue,
            'aic': model.aic,
            'bic': model.bic,
            'mse_resid': model.mse_resid,
            'coefficients': dict(zip(X.columns, model.params)),
            'pvalues': dict(zip(X.columns, model.pvalues)),
            'conf_int': model.conf_int().to_dict(),
            'significant_features': self.get_significant_features(model, X.columns)
        }
        
        logger.info(f"Model R-squared: {stats['r_squared']:.4f}")
        logger.info(f"Model Adjusted R-squared: {stats['adj_r_squared']:.4f}")
        logger.info(f"Significant features at {self.significance_level} level: {stats['significant_features']}")
        
        return model, stats
    
    def train_sklearn_linear_regression(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, Dict]:
        """
        Train linear regression model using scikit-learn.
        
        Args:
            X: Features DataFrame
            y: Target Series
            
        Returns:
            Tuple of (fitted model, model statistics)
        """
        logger.info("Training Linear Regression model using scikit-learn")
        
        # Fit the model
        model = LinearRegression()
        model.fit(X, y)
        
        # Make predictions for statistics
        y_pred = model.predict(X)
        
        # Calculate statistics
        stats = {
            'r_squared': r2_score(y, y_pred),
            'mse': mean_squared_error(y, y_pred),
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'coefficients': dict(zip(X.columns, model.coef_)),
            'intercept': model.intercept_
        }
        
        logger.info(f"Model R-squared: {stats['r_squared']:.4f}")
        logger.info(f"Model RMSE: {stats['rmse']:.4f}")
        
        return model, stats
    
    
    def get_significant_features(self, model: Any, feature_names: list) -> list:
        """
        Get features that are significant at the specified level.
        
        Args:
            model: Fitted statsmodels model
            feature_names: List of feature names
            
        Returns:
            List of significant feature names
        """
        significant_features = []
        for feature in feature_names:
            if feature in model.pvalues and model.pvalues[feature] <= self.significance_level:
                significant_features.append(feature)
        
        return significant_features
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, Dict]:
        """
        Train the specified model type.
        
        Args:
            X: Features DataFrame
            y: Target Series
            
        Returns:
            Tuple of (fitted model, model statistics)
        """
        logger.info(f"Training linear regression model")
        return self.train_sklearn_linear_regression(X, y)

    
    
    def feature_selection_by_significance(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, list]:
        """
        Perform feature selection based on statistical significance.
        
        Args:
            X: Features DataFrame
            y: Target Series
            
        Returns:
            Tuple of (selected features DataFrame, list of selected feature names)
        """
        logger.info(f"Performing feature selection at significance level {self.significance_level}")
        
        # Train initial model to get p-values
        model, stats = self.train_statsmodels_ols(X, y)
        
        # Get significant features
        significant_features = stats['significant_features']
        
        if not significant_features:
            logger.warning("No significant features found. Using all features.")
            return X, list(X.columns)
        
        # Select only significant features
        X_selected = X[significant_features]
        
        logger.info(f"Selected {len(significant_features)} significant features: {significant_features}")
        return X_selected, significant_features
    
    def save_model(self, model: Any, output_path: str, model_stats: Dict = None):
        """
        Save the trained model and statistics.
        
        Args:
            model: Trained model
            output_path: Path to save the model
            model_stats: Model statistics dictionary
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save model
        if hasattr(model, 'save'):
            # For statsmodels
            model.save(output_path)
        else:
            # For sklearn models
            joblib.dump(model, output_path)
        
        logger.info(f"Model saved to {output_path}")
        
        # Save model statistics
        if model_stats:
            stats_path = output_path.replace('.joblib', '_stats.json').replace('.pkl', '_stats.json')
            
            # Convert numpy types to native Python types for JSON serialization
            serializable_stats = self._make_json_serializable(model_stats)
            
            with open(stats_path, 'w') as f:
                json.dump(serializable_stats, f, indent=2)
            
            logger.info(f"Model statistics saved to {stats_path}")
    
    def _make_json_serializable(self, obj):
        """
        Convert numpy types to native Python types for JSON serialization.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON-serializable object
        """
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj


def main():
    """
    Main function for standalone execution.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument('--features-path', type=str, required=True, help='Features CSV path')
    parser.add_argument('--target-path', type=str, required=True, help='Target CSV path')
    parser.add_argument('--model-output-path', type=str, required=True, help='Model output path')
    parser.add_argument('--config-path', type=str, help='Configuration file path')
    parser.add_argument('--feature-selection', action='store_true', help='Perform feature selection')

    args = parser.parse_args()

    def install_package(package, version):
        subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package}=={version}"])

    install_package('statsmodels','0.14.0')
    
    # Initialize trainer
    trainer = ModelTrainer(args.config_path)
    
    # Load data
    X = pd.read_csv(args.features_path)
    y = pd.read_csv(args.target_path).squeeze()  # Convert to Series
    
    logger.info(f"Loaded features: {X.shape}, target: {y.shape}")
    
    # Perform feature selection if requested
    if args.feature_selection:
        X, selected_features = trainer.feature_selection_by_significance(X, y)
        logger.info(f"Selected features: {selected_features}")
    
    # Train model only (evaluation is handled separately with test data)
    model, train_stats = trainer.train_model(X, y)
    
    # Save model
    trainer.save_model(model, args.model_output_path, trainer.model_stats)
    
    logger.info("Model training completed successfully")


if __name__ == "__main__":
    main()
