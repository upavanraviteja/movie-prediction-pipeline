"""
Inference module for movie box office prediction pipeline.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Union
import yaml
import joblib
import json
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MovieBoxOfficePredictor:
    """
    Movie box office prediction inference class.
    """
    
    def __init__(self, model_path: str = None, config_path: str = None, 
                 artifacts_path: str = None):
        """
        Initialize the predictor with model and configuration.
        
        Args:
            model_path: Path to the trained model
            config_path: Path to configuration file
            artifacts_path: Path to feature engineering artifacts
        """
        if config_path:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        else:
            self.config = {}
        
        self.model = None
        self.scaler = None
        self.pca = None
        self.feature_names = None
        self.model_stats = None
        
        # Load model and artifacts if paths provided
        if model_path:
            self.load_model(model_path)
        
        if artifacts_path:
            self.load_artifacts(artifacts_path)
    
    def load_model(self, model_path: str):
        """
        Load the trained model.
        
        Args:
            model_path: Path to the saved model
        """
        try:
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
            
            # Try to load model statistics
            stats_path = model_path.replace('.joblib', '_stats.json').replace('.pkl', '_stats.json')
            if os.path.exists(stats_path):
                with open(stats_path, 'r') as f:
                    self.model_stats = json.load(f)
                logger.info(f"Model statistics loaded from {stats_path}")
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def load_artifacts(self, artifacts_path: str):
        """
        Load feature engineering artifacts.
        
        Args:
            artifacts_path: Path to artifacts directory
        """
        try:
            scaler_path = os.path.join(artifacts_path, 'scaler.joblib')
            pca_path = os.path.join(artifacts_path, 'pca.joblib')
            
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info(f"Scaler loaded from {scaler_path}")
            
            if os.path.exists(pca_path):
                self.pca = joblib.load(pca_path)
                logger.info(f"PCA loaded from {pca_path}")
                
        except Exception as e:
            logger.error(f"Error loading artifacts: {str(e)}")
            raise
    
    def preprocess_input(self, input_data: Union[Dict, pd.DataFrame]) -> pd.DataFrame:
        """
        Preprocess input data for prediction.
        
        Args:
            input_data: Input data as dictionary or DataFrame
            
        Returns:
            Preprocessed DataFrame ready for prediction
        """
        logger.info("Preprocessing input data")
        
        # Convert to DataFrame if dictionary
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        # Apply log transformations based on configuration
        log_transform_columns = self.config.get('feature_engineering', {}).get(
            'log_transform_columns', ['box', 'budget', 'addict', 'cmngsoon', 'fandango']
        )
        
        for col in log_transform_columns:
            if col in df.columns:
                # Check if column has positive values
                if (df[col] > 0).all():
                    new_col_name = f'log_{col}'
                    df[new_col_name] = np.log(df[col])
                    logger.info(f"Applied log transformation to {col} -> {new_col_name}")
        
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for prediction following the training pipeline.
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            Feature DataFrame ready for model prediction
        """
        logger.info("Creating features for prediction")
        
        # Get target and drop columns from config
        target_column = self.config.get('model', {}).get('target_column', 'log_box')
        drop_columns = self.config.get('model', {}).get('drop_columns', [])
        
        # Prepare features similar to training
        X = df.copy()
        
        # Drop target column and specified columns if they exist
        columns_to_drop = [target_column] + drop_columns
        columns_to_drop = [col for col in columns_to_drop if col in X.columns]
        if columns_to_drop:
            X = X.drop(columns=columns_to_drop)
        
        # Create traditional features (exclude buzz variables)
        buzz_vars = ['addict', 'cmngsoon', 'fandango', 'cntwait3', 
                    'log_addict', 'log_cmngsoon', 'log_fandango']
        
        traditional_features = X.copy()
        buzz_cols_to_drop = [col for col in buzz_vars if col in traditional_features.columns]
        traditional_features = traditional_features.drop(columns=buzz_cols_to_drop)
        
        # Add constant term for regression
        traditional_features['const'] = 1.0
        
        # Create buzz features for PCA if available
        available_buzz_vars = [col for col in buzz_vars if col in X.columns]
        
        if available_buzz_vars and self.scaler is not None and self.pca is not None:
            logger.info("Applying PCA to buzz features")
            
            buzz_features = X[available_buzz_vars]
            
            # Apply same preprocessing as training
            X_scaled = self.scaler.transform(buzz_features)
            sample_std = np.std(X_scaled, axis=0, ddof=1)
            X_scaled = X_scaled / sample_std
            
            # Apply PCA
            X_pca = self.pca.transform(X_scaled)
            
            # Create PCA DataFrame
            pc_columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
            pca_df = pd.DataFrame(X_pca, columns=pc_columns, index=traditional_features.index)
            
            # Combine traditional features with PCA components
            combined_features = pd.concat([traditional_features, pca_df], axis=1)
            
        else:
            logger.warning("No buzz features or PCA artifacts available, using only traditional features")
            combined_features = traditional_features
        
        logger.info(f"Created features with shape: {combined_features.shape}")
        logger.info(f"Feature columns: {list(combined_features.columns)}")
        
        return combined_features
    
    def predict(self, input_data: Union[Dict, pd.DataFrame]) -> Dict:
        """
        Make prediction on input data.
        
        Args:
            input_data: Input data as dictionary or DataFrame
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load a model first.")
        
        logger.info("Making prediction")
        
        # Preprocess input
        df_processed = self.preprocess_input(input_data)
        
        # Create features
        X_features = self.create_features(df_processed)
        
        # Make prediction
        prediction = self.model.predict(X_features)
        
        # Convert back from log scale if target was log-transformed
        target_column = self.config.get('model', {}).get('target_column', 'log_box')
        if 'log_' in target_column:
            # Convert back to original scale
            original_prediction = np.exp(prediction)
            
            result = {
                'log_prediction': prediction.tolist() if hasattr(prediction, 'tolist') else [prediction],
                'prediction': original_prediction.tolist() if hasattr(original_prediction, 'tolist') else [original_prediction],
                'prediction_type': 'box_office_revenue',
                'model_info': {
                    'target_column': target_column,
                    'n_features': X_features.shape[1],
                    'feature_names': list(X_features.columns)
                }
            }
        else:
            result = {
                'prediction': prediction.tolist() if hasattr(prediction, 'tolist') else [prediction],
                'prediction_type': target_column,
                'model_info': {
                    'target_column': target_column,
                    'n_features': X_features.shape[1],
                    'feature_names': list(X_features.columns)
                }
            }
        
        # Add model statistics if available
        if self.model_stats:
            result['model_stats'] = {
                'r_squared': self.model_stats.get('r_squared'),
                'adj_r_squared': self.model_stats.get('adj_r_squared'),
                'test_r2': self.model_stats.get('test_r2')
            }
        
        logger.info(f"Prediction completed: {result['prediction']}")
        return result
    
    def predict_batch(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Make batch predictions on multiple samples.
        
        Args:
            input_data: DataFrame with multiple samples
            
        Returns:
            DataFrame with predictions
        """
        logger.info(f"Making batch predictions for {len(input_data)} samples")
        
        # Preprocess input
        df_processed = self.preprocess_input(input_data)
        
        # Create features
        X_features = self.create_features(df_processed)
        
        # Make predictions
        predictions = self.model.predict(X_features)
        
        # Create results DataFrame
        results_df = input_data.copy()
        
        target_column = self.config.get('model', {}).get('target_column', 'log_box')
        if 'log_' in target_column:
            results_df['log_prediction'] = predictions
            results_df['prediction'] = np.exp(predictions)
        else:
            results_df['prediction'] = predictions
        
        logger.info("Batch predictions completed")
        return results_df
    
    
    def validate_input(self, input_data: Union[Dict, pd.DataFrame]) -> bool:
        """
        Validate input data format and required fields.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        required_fields = ['G', 'PG', 'PG13', 'starpowr', 'sequel', 'action', 
                          'comedy', 'animated', 'horror', 'log_budget']
        
        if isinstance(input_data, dict):
            missing_fields = [field for field in required_fields if field not in input_data]
        else:
            missing_fields = [field for field in required_fields if field not in input_data.columns]
        
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        return True


def main():
    """
    Main function for standalone inference.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Movie Box Office Prediction Inference')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--config-path', type=str, help='Configuration file path')
    parser.add_argument('--artifacts-path', type=str, help='Feature engineering artifacts path')
    parser.add_argument('--input-path', type=str, help='Input CSV file path for batch prediction')
    parser.add_argument('--output-path', type=str, help='Output CSV file path for batch prediction')
    parser.add_argument('--single-prediction', action='store_true', help='Make single prediction from CLI args')
    
    # Add movie feature arguments for single prediction
    parser.add_argument('--G', type=int, default=0, help='G rating (0 or 1)')
    parser.add_argument('--PG', type=int, default=0, help='PG rating (0 or 1)')
    parser.add_argument('--PG13', type=int, default=0, help='PG13 rating (0 or 1)')
    parser.add_argument('--starpowr', type=float, default=0.0, help='Star power rating')
    parser.add_argument('--sequel', type=int, default=0, help='Sequel (0 or 1)')
    parser.add_argument('--action', type=int, default=0, help='Action genre (0 or 1)')
    parser.add_argument('--comedy', type=int, default=0, help='Comedy genre (0 or 1)')
    parser.add_argument('--animated', type=int, default=0, help='Animated genre (0 or 1)')
    parser.add_argument('--horror', type=int, default=0, help='Horror genre (0 or 1)')
    parser.add_argument('--log-budget', type=float, required=False, help='Log budget')
    parser.add_argument('--budget', type=float, help='Budget (will be log-transformed)')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = MovieBoxOfficePredictor(
        model_path=args.model_path,
        config_path=args.config_path,
        artifacts_path=args.artifacts_path
    )
    
    if args.single_prediction:
        # Single prediction from CLI arguments
        input_data = {
            'G': args.G,
            'PG': args.PG,
            'PG13': args.PG13,
            'starpowr': args.starpowr,
            'sequel': args.sequel,
            'action': args.action,
            'comedy': args.comedy,
            'animated': args.animated,
            'horror': args.horror
        }
        
        if args.log_budget is not None:
            input_data['log_budget'] = args.log_budget
        elif args.budget is not None:
            input_data['budget'] = args.budget
        else:
            raise ValueError("Either --log-budget or --budget must be provided")
        
        # Make prediction
        result = predictor.predict(input_data)
        print(json.dumps(result, indent=2))
        
    elif args.input_path and args.output_path:
        # Batch prediction
        input_df = pd.read_csv(args.input_path)
        results_df = predictor.predict_batch(input_df)
        
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        results_df.to_csv(args.output_path, index=False)
        logger.info(f"Batch predictions saved to {args.output_path}")
        
    else:
        logger.error("Either --single-prediction or both --input-path and --output-path must be provided")


if __name__ == "__main__":
    main()
