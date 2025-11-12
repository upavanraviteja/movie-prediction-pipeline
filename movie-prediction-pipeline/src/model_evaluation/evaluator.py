"""
Model evaluation module for movie box office prediction pipeline.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Model evaluation class for movie box office prediction.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the model evaluator with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        if config_path:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        else:
            self.config = {}
        
        self.significance_level = self.config.get('model', {}).get('significance_level', 0.1)
    
    def load_model(self, model_path: str) -> Any:
        """
        Load a trained model.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded model
        """
        try:
            model = joblib.load(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def load_model_stats(self, stats_path: str) -> Dict:
        """
        Load model statistics.
        
        Args:
            stats_path: Path to the model statistics JSON file
            
        Returns:
            Dictionary with model statistics
        """
        try:
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            logger.info(f"Model statistics loaded from {stats_path}")
            return stats
        except Exception as e:
            logger.error(f"Error loading model statistics: {str(e)}")
            raise
    
    def calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calculate comprehensive regression metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            
        Returns:
            Dictionary with regression metrics
        """
        metrics = {
            'r2_score': r2_score(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'residual_std': np.std(y_true - y_pred),
            'n_samples': len(y_true)
        }
        
        return metrics
    
    def evaluate_model_performance(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[Dict, pd.Series]:
        """
        Evaluate model performance on test data.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target values
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating model performance")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = self.calculate_regression_metrics(y_test.values, y_pred)
        
        logger.info(f"Model Performance Metrics:")
        logger.info(f"  R² Score: {metrics['r2_score']:.4f}")
        logger.info(f"  RMSE: {metrics['rmse']:.4f}")
        logger.info(f"  MAE: {metrics['mae']:.4f}")
        logger.info(f"  MAPE: {metrics['mape']:.2f}%")
        
        return metrics, y_pred

    
    def create_evaluation_plots(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              output_dir: str, model_name: str = "model"):
        """
        Create evaluation plots.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            output_dir: Directory to save plots
            model_name: Name of the model for plot titles
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Actual vs Predicted scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_pred, alpha=0.6, s=50)
        
        # Perfect prediction line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{model_name}: Actual vs Predicted Values')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = os.path.join(output_dir, f'{model_name}_actual_vs_predicted.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Actual vs Predicted plot saved to {plot_path}")
        
        # 2. Residuals plot
        residuals = y_true - y_pred
        
        plt.figure(figsize=(12, 5))
        
        # Residuals vs Predicted
        plt.subplot(1, 2, 1)
        plt.scatter(y_pred, residuals, alpha=0.6, s=50)
        plt.axhline(y=0, color='r', linestyle='--', lw=2)
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title(f'{model_name}: Residuals vs Predicted')
        plt.grid(True, alpha=0.3)
        
        # Residuals histogram
        plt.subplot(1, 2, 2)
        plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title(f'{model_name}: Residuals Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'{model_name}_residuals_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Residuals analysis plot saved to {plot_path}")
        
        # 3. Q-Q plot for residuals normality
        from scipy import stats
        
        plt.figure(figsize=(8, 6))
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title(f'{model_name}: Q-Q Plot of Residuals')
        plt.grid(True, alpha=0.3)
        
        plot_path = os.path.join(output_dir, f'{model_name}_qq_plot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Q-Q plot saved to {plot_path}")
    
    
    def generate_evaluation_report(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                                 model_stats: Dict = None, output_dir: str = None,
                                 model_name: str = "model") -> Dict:
        """
        Generate comprehensive evaluation report.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target values
            model_stats: Model statistics dictionary
            output_dir: Directory to save plots and reports
            model_name: Name of the model
            
        Returns:
            Dictionary with complete evaluation results
        """
        logger.info(f"Generating comprehensive evaluation report for {model_name}")

        # Calculate performance metrics
        performance_metrics, y_pred = self.evaluate_model_performance(model, X_test, y_test)
        
        # Create evaluation report
        evaluation_report = {
            'model_name': model_name,
            'performance_metrics': performance_metrics,
            'model_stats': model_stats or {}
        }
        
        # Create plots if output directory is provided
        if output_dir:
            self.create_evaluation_plots(y_test.values, y_pred, output_dir, model_name)
            
            # Save evaluation report
            report_path = os.path.join(output_dir, f'{model_name}_evaluation_report.json')
            with open(report_path, 'w') as f:
                json.dump(evaluation_report, f, indent=2, default=str)
            logger.info(f"Evaluation report saved to {report_path}")
        
        logger.info("Evaluation report generation completed")
        return evaluation_report


def main():
    """
    Main function for standalone execution.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Model Evaluation')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--features-path', type=str, required=True, help='Test features CSV path')
    parser.add_argument('--target-path', type=str, required=True, help='Test target CSV path')
    parser.add_argument('--stats-path', type=str, help='Model statistics JSON path')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for reports')
    parser.add_argument('--config-path', type=str, help='Configuration file path')
    parser.add_argument('--model-name', type=str, default='model', help='Model name for reports')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.config_path)
    
    # Load model
    model = evaluator.load_model(args.model_path)
    
    # Load model statistics if provided
    model_stats = None
    if args.stats_path and os.path.exists(args.stats_path):
        model_stats = evaluator.load_model_stats(args.stats_path)
    
    # Load test data
    X_test = pd.read_csv(args.features_path)
    y_test = pd.read_csv(args.target_path).squeeze()
    
    logger.info(f"Loaded test data: X={X_test.shape}, y={y_test.shape}")
    
    # Generate evaluation report
    evaluation_report = evaluator.generate_evaluation_report(
        model, X_test, y_test, model_stats, args.output_dir, args.model_name
    )
    
    logger.info("Model evaluation completed successfully")


if __name__ == "__main__":
    main()
