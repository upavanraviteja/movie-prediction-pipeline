"""
SageMaker Pipeline for Movie Box Office Prediction MLOps.
"""

import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.parameters import ParameterInteger, ParameterString, ParameterFloat
from sagemaker.workflow.pipeline_context import PipelineSession

from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.drift_check_baselines import DriftCheckBaselines

import json
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MoviePredictionPipeline:
    """
    SageMaker Pipeline for Movie Box Office Prediction.
    """
    
    def __init__(self, config_path: str, role: str = None, region: str = None):
        """
        Initialize the pipeline.
        
        Args:
            config_path: Path to configuration file
            role: SageMaker execution role
            region: AWS region
        """
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = json.load(file)
        
        # Initialize SageMaker session
        self.region = region or boto3.Session().region_name
        self.sagemaker_session = PipelineSession()
        
        # Set role
        self.role = role or self.config.get('sagemaker', {}).get('role')
        if not self.role:
            self.role = sagemaker.get_execution_role()
        
        # Pipeline configuration
        self.pipeline_name = self.config.get('pipeline', {}).get('name', 'movie-box-office-prediction')
        self.pipeline_description = self.config.get('pipeline', {}).get('description', 'MLOps pipeline for predicting movie box office revenue')
        
        # SageMaker configuration
        self.instance_type = self.config.get('sagemaker', {}).get('instance_type', 'ml.m5.large')
        self.instance_count = self.config.get('sagemaker', {}).get('instance_count', 1)
        self.framework_version = self.config.get('sagemaker', {}).get('framework_version', '1.0-1')
        self.py_version = self.config.get('sagemaker', {}).get('py_version', 'py3')
        
        # S3 bucket for pipeline artifacts
        self.bucket = sagemaker.Session().default_bucket()
        self.prefix = 'movie-prediction-pipeline'
        
        logger.info(f"Initialized pipeline with bucket: {self.bucket}, role: {self.role}")
    
    def create_parameters(self):
        """
        Create pipeline parameters.
        
        Returns:
            Dictionary of pipeline parameters
        """
        parameters = {
            'input_data': ParameterString(
                name="InputData",
                default_value=f"s3://{self.bucket}/{self.prefix}/data/boxOffice.csv"
            ),
            'model_approval_status': ParameterString(
                name="ModelApprovalStatus",
                default_value="PendingManualApproval"
            ),
            'processing_instance_type': ParameterString(
                name="ProcessingInstanceType",
                default_value=self.instance_type
            ),
            'training_instance_type': ParameterString(
                name="TrainingInstanceType",
                default_value=self.instance_type
            ),
            'model_performance_threshold': ParameterFloat(
                name="ModelPerformanceThreshold",
                default_value=0.6
            )
        }
        
        return parameters
    
    def create_preprocessing_step(self, parameters):
        """
        Create data preprocessing step using sklearn_processor.run().
        
        Args:
            parameters: Pipeline parameters
            
        Returns:
            ProcessingStep for data preprocessing
        """
        # Create SKLearn processor
        sklearn_processor = SKLearnProcessor(
            framework_version=self.framework_version,
            instance_type=parameters['processing_instance_type'],
            instance_count=self.instance_count,
            base_job_name="movie-prediction-preprocessing",
            sagemaker_session=self.sagemaker_session,
            role=self.role,
        )
        
        # Use sklearn_processor.run() as specified
        processor_args = sklearn_processor.run(
            inputs=[
                ProcessingInput(source=parameters['input_data'], destination="/opt/ml/processing/preprocessing/input"),
                ProcessingInput(source=f"s3://{self.bucket}/{self.prefix}/config/config.json", destination="/opt/ml/processing/preprocessing/config"),
                ProcessingInput(source=f"s3://{self.bucket}/{self.prefix}/requirements.txt", destination="/opt/ml/processing/preprocessing/requirements")
            ],
            outputs=[
                ProcessingOutput(output_name="output", source="/opt/ml/processing/preprocessing/output",
                    destination=f"s3://{self.bucket}/{self.prefix}/output"
                    ),
            ],
            code="src/data_preprocessing/preprocessor.py",
            arguments=[
                "--input-path", "/opt/ml/processing/preprocessing/input/boxOffice.csv",
                "--output-path", "/opt/ml/processing/preprocessing/output/preppedBoxOffice.csv",
                "--config-path", "/opt/ml/processing/preprocessing/config/config.json",
            ]
        )
        
        # Create ProcessingStep using the processor_args
        step_preprocessing = ProcessingStep(
            name="MoviePredictionPreprocessing",
            step_args=processor_args
        )
        
        return step_preprocessing
    
    def create_feature_engineering_step(self, step_preprocessing, parameters):
        """
        Create feature engineering step using sklearn_processor.run().
        
        Args:
            step_preprocessing: Previous preprocessing step
            parameters: Pipeline parameters
            
        Returns:
            ProcessingStep for feature engineering
        """
        # Create SKLearn processor
        sklearn_processor = SKLearnProcessor(
            framework_version=self.framework_version,
            instance_type=parameters['processing_instance_type'],
            instance_count=self.instance_count,
            base_job_name="movie-prediction-feature-engineering",
            sagemaker_session=self.sagemaker_session,
            role=self.role,
        )
        
        # Use sklearn_processor.run() for feature engineering
        processor_args = sklearn_processor.run(
            inputs=[
                ProcessingInput(
                    source=step_preprocessing.properties.ProcessingOutputConfig.Outputs[
                        "output"
                    ].S3Output.S3Uri,
                    destination="/opt/ml/processing/input"
                ),
                ProcessingInput(
                    source=f"s3://{self.bucket}/{self.prefix}/config/config.json",
                    destination="/opt/ml/processing/config"
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="train_data",
                    source="/opt/ml/processing/output/train",
                    destination=f"s3://{self.bucket}/{self.prefix}/train"
                ),
                ProcessingOutput(
                    output_name="test_data",
                    source="/opt/ml/processing/output/test",
                    destination=f"s3://{self.bucket}/{self.prefix}/test"
                ),
                ProcessingOutput(
                    output_name="feature_artifacts",
                    source="/opt/ml/processing/output/artifacts",
                    destination=f"s3://{self.bucket}/{self.prefix}/artifacts"
                )
            ],
            code="src/feature_engineering/feature_engineer.py",
            arguments=[
                "--input-path", "/opt/ml/processing/input/preppedBoxOffice.csv",
                "--train-features-path", "/opt/ml/processing/output/train/train_features.csv",
                "--train-target-path", "/opt/ml/processing/output/train/train_target.csv",
                "--test-features-path", "/opt/ml/processing/output/test/test_features.csv",
                "--test-target-path", "/opt/ml/processing/output/test/test_target.csv",
                "--artifacts-path", "/opt/ml/processing/output/artifacts",
                "--config-path", "/opt/ml/processing/config/config.json"
            ]
        )
        
        # Create ProcessingStep using the processor_args
        step_feature_engineering = ProcessingStep(
            name="MoviePredictionFeatureEngineering",
            step_args=processor_args
        )
        
        return step_feature_engineering
    
    def create_training_step(self, step_feature_engineering, parameters):
        """
        Create model training step.
        
        Args:
            step_feature_engineering: Previous feature engineering step
            parameters: Pipeline parameters
            
        Returns:
            TrainingStep for model training
        """
        # Create SKLearn estimator
        sklearn_estimator = SKLearn(
            entry_point="trainer.py",
            source_dir="src/model_training",
            framework_version=self.framework_version,
            py_version=self.py_version,
            instance_type=parameters['training_instance_type'],
            instance_count=self.instance_count,
            role=self.role,
            sagemaker_session=self.sagemaker_session,
            dependencies=["config/config.json"],
            hyperparameters={
                "features-path": "/opt/ml/input/data/train/train_features.csv",
                "target-path": "/opt/ml/input/data/train/train_target.csv",
                "model-output-path": "/opt/ml/model/model.joblib",
                "config-path": "/opt/ml/code/config/config.json",
                "feature-selection": True
            }
        )
        
        # Define training step
        step_training = TrainingStep(
            name="MoviePredictionTraining",
            estimator=sklearn_estimator,
            inputs={
                "train": TrainingInput(
                    s3_data=step_feature_engineering.properties.ProcessingOutputConfig.Outputs[
                        "train_data"
                    ].S3Output.S3Uri,
                    content_type="text/csv"
                )
            }
        )
        
        return step_training
    
    def create_evaluation_step(self, step_training, step_feature_engineering, parameters):
        """
        Create model evaluation step using sklearn_processor.run().
        
        Args:
            step_training: Previous training step
            step_feature_engineering: Feature engineering step
            parameters: Pipeline parameters
            
        Returns:
            ProcessingStep for model evaluation
        """
        from sagemaker.workflow.properties import PropertyFile
        
        # Create SKLearn processor for evaluation
        sklearn_processor = SKLearnProcessor(
            framework_version=self.framework_version,
            instance_type=parameters['processing_instance_type'],
            instance_count=self.instance_count,
            base_job_name="movie-prediction-evaluation",
            sagemaker_session=self.sagemaker_session,
            role=self.role,
        )
        
        # Define property file for evaluation results
        evaluation_report = PropertyFile(
            name="EvaluationReport",
            output_name="evaluation_report",
            path="model_evaluation_report.json"
        )
        
        # Use sklearn_processor.run() for evaluation
        processor_args = sklearn_processor.run(
            inputs=[
                ProcessingInput(
                    source=step_training.properties.ModelArtifacts.S3ModelArtifacts,
                    destination="/opt/ml/processing/model"
                ),
                ProcessingInput(
                    source=step_feature_engineering.properties.ProcessingOutputConfig.Outputs[
                        "test_data"
                    ].S3Output.S3Uri,
                    destination="/opt/ml/processing/test"
                ),
                ProcessingInput(
                    source=f"s3://{self.bucket}/{self.prefix}/config/config.json",
                    destination="/opt/ml/processing/config"
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="evaluation_report",
                    source="/opt/ml/processing/output",
                    destination=f"s3://{self.bucket}/{self.prefix}/evaluation"
                )
            ],
            code="src/model_evaluation/evaluator.py",
            arguments=[
                "--model-path", "/opt/ml/processing/model/model.joblib",
                "--features-path", "/opt/ml/processing/test/test_features.csv",
                "--target-path", "/opt/ml/processing/test/test_target.csv",
                "--output-dir", "/opt/ml/processing/output",
                "--config-path", "/opt/ml/processing/config/config.json",
                "--model-name", "movie-prediction-model"
            ]
        )
        
        # Create ProcessingStep using the processor_args with property files
        step_evaluation = ProcessingStep(
            name="MoviePredictionEvaluation",
            step_args=processor_args,
            property_files=[evaluation_report]
        )
        
        return step_evaluation, evaluation_report
    
    def create_model_registration_step(self, step_training, step_evaluation, parameters):
        """
        Create model registration step.
        
        Args:
            step_training: Training step
            step_evaluation: Evaluation step
            parameters: Pipeline parameters
            
        Returns:
            RegisterModel step
        """
        from sagemaker.workflow.functions import Join
        
        # Create model metrics using Join function for proper pipeline variable handling
        model_metrics = ModelMetrics(
            model_statistics=MetricsSource(
                s3_uri=Join(
                    on="/",
                    values=[
                        step_evaluation.properties.ProcessingOutputConfig.Outputs['evaluation_report'].S3Output.S3Uri,
                        "model_evaluation_report.json"
                    ]
                ),
                content_type="application/json"
            )
        )
        
        # Create drift check baselines using Join function
        drift_check_baselines = DriftCheckBaselines(
            model_statistics=MetricsSource(
                s3_uri=Join(
                    on="/",
                    values=[
                        step_evaluation.properties.ProcessingOutputConfig.Outputs['evaluation_report'].S3Output.S3Uri,
                        "model_evaluation_report.json"
                    ]
                ),
                content_type="application/json"
            )
        )
        
        # Register model step
        step_register = RegisterModel(
            name="MoviePredictionRegisterModel",
            estimator=step_training.estimator,
            model_data=step_training.properties.ModelArtifacts.S3ModelArtifacts,
            content_types=["text/csv"],
            response_types=["text/csv"],
            inference_instances=["ml.t2.medium", "ml.m5.large"],
            transform_instances=["ml.m5.large"],
            model_package_group_name="movie-prediction-model-group",
            approval_status=parameters['model_approval_status'],
            model_metrics=model_metrics,
            drift_check_baselines=drift_check_baselines
        )
        
        return step_register
    
    def create_condition_step(self, step_evaluation, evaluation_report, step_register, parameters):
        """
        Create conditional step for model registration based on performance.
        
        Args:
            step_evaluation: Evaluation step
            evaluation_report: PropertyFile for evaluation results
            step_register: Model registration step
            parameters: Pipeline parameters
            
        Returns:
            ConditionStep for conditional model registration
        """
        # Define condition for model performance
        cond_gte = ConditionGreaterThanOrEqualTo(
            left=JsonGet(
                step_name=step_evaluation.name,
                property_file=evaluation_report,
                json_path="performance_metrics.test_r2"
            ),
            right=parameters['model_performance_threshold']
        )
        
        # Create condition step
        step_condition = ConditionStep(
            name="MoviePredictionCondition",
            conditions=[cond_gte],
            if_steps=[step_register],
            else_steps=[]
        )
        
        return step_condition
    
    def create_pipeline(self):
        """
        Create the complete SageMaker pipeline.
        
        Returns:
            SageMaker Pipeline object
        """
        logger.info("Creating SageMaker pipeline")
        
        # Create parameters
        parameters = self.create_parameters()
        
        # Create pipeline steps
        step_preprocessing = self.create_preprocessing_step(parameters)
        step_feature_engineering = self.create_feature_engineering_step(step_preprocessing, parameters)
        step_training = self.create_training_step(step_feature_engineering, parameters)
        step_evaluation, evaluation_report = self.create_evaluation_step(step_training, step_feature_engineering, parameters)
        
        # Create pipeline
        pipeline = Pipeline(
            name=self.pipeline_name,
            parameters=list(parameters.values()),
            steps=[
                step_preprocessing,
                step_feature_engineering,
                step_training,
                step_evaluation,
            ],
            sagemaker_session=self.sagemaker_session
        )
        
        logger.info(f"Pipeline created: {self.pipeline_name}")
        return pipeline
    
    def deploy_pipeline(self, pipeline):
        """
        Deploy the pipeline to SageMaker.
        
        Args:
            pipeline: SageMaker Pipeline object
            
        Returns:
            Pipeline execution response
        """
        logger.info("Deploying pipeline to SageMaker")
        
        # Upsert pipeline
        pipeline.upsert(role_arn=self.role)
        
        logger.info(f"Pipeline {self.pipeline_name} deployed successfully")
        return pipeline
    
    def execute_pipeline(self, pipeline, parameters=None):
        """
        Execute the pipeline.
        
        Args:
            pipeline: SageMaker Pipeline object
            parameters: Optional parameters for execution
            
        Returns:
            Pipeline execution object
        """
        logger.info("Starting pipeline execution")
        
        execution = pipeline.start(parameters=parameters)
        
        logger.info(f"Pipeline execution started: {execution.arn}")
        return execution


def main():
    """
    Main function to create and deploy the pipeline.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='SageMaker Pipeline for Movie Prediction')
    parser.add_argument('--config-path', type=str, default='config.json', help='Configuration file path')
    parser.add_argument('--role', type=str, help='SageMaker execution role ARN')
    parser.add_argument('--region', type=str, help='AWS region')
    parser.add_argument('--deploy', action='store_true', help='Deploy the pipeline')
    parser.add_argument('--execute', action='store_true', help='Execute the pipeline after deployment')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline_manager = MoviePredictionPipeline(
        config_path=args.config_path,
        role=args.role,
        region=args.region
    )
    
    # Create pipeline
    pipeline = pipeline_manager.create_pipeline()

    if args.deploy:
        # Deploy pipeline
        deployed_pipeline = pipeline_manager.deploy_pipeline(pipeline)
    
        if args.execute:
            # Execute pipeline
            execution = pipeline_manager.execute_pipeline(deployed_pipeline)
            print(f"Pipeline execution started: {execution.arn}")



if __name__ == "__main__":
    main()
