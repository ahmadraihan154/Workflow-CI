import argparse
import os
import numpy as np
import pandas as pd
import joblib
import mlflow
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings(action='ignore')

def train_model(data_path, n_estimators, output_path):
    # Load Data
    transformer = joblib.load(os.path.join(data_path, 'power_transformers.joblib'))
    price_transformer = transformer['price']
    X_train = pd.read_csv(os.path.join(data_path, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(data_path, 'y_train.csv'))
    X_test = pd.read_csv(os.path.join(data_path, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(data_path, 'y_test.csv'))
    y_test = price_transformer.inverse_transform(y_test.to_numpy().reshape(-1, 1))

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Set experiment in MLflow (ensure experiment exists in MLflow)
    mlflow.set_experiment('base-model_experiment_2')  # Replace with your experiment name

    with mlflow.start_run(run_name='LGBM_Base_2'):
        # Define model
        model = LGBMRegressor(n_estimators=n_estimators)
        model.fit(X_train, y_train)  
        
        # Predictions
        y_pred_transform = model.predict(X_test)
        y_pred = price_transformer.inverse_transform(y_pred_transform.reshape(-1, 1))
        
        # Calculate metrics
        r2_skor = r2_score(y_test, y_pred)
        rmse_skor = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Log parameters and metrics to MLflow
        mlflow.log_params(model.get_params())
        mlflow.log_metric("RMSE", rmse_skor)    
        mlflow.log_metric("R2", r2_skor)
        mlflow.sklearn.log_model(model, artifact_path="model")
        
        # Save model for GitHub Actions artifact
        model_file = os.path.join(output_path, 'lgbm_model.joblib')
        joblib.dump(model, model_file)
        
        # Save transformer for GitHub Actions artifact
        transformer_file = os.path.join(output_path, 'transformer.joblib')
        joblib.dump(transformer, transformer_file)
        
        # Save metrics to a file for GitHub Actions artifact
        metrics = {
            'r2_score': r2_skor,
            'rmse': rmse_skor
        }
        metrics_file = os.path.join(output_path, 'metrics.joblib')
        joblib.dump(metrics, metrics_file)

        # Print metrics
        print(f'R2 Score : {r2_skor}')
        print(f'RMSE : {rmse_skor}')
        print(f'Model saved to: {model_file}')
    
    # Ensure the run ends
    mlflow.end_run()

    return model, r2_skor, rmse_skor

if __name__ == '__main__':
    # Argument parser for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='diamond_preprocessing',
                        help='Path to preprocessed data directory')
    parser.add_argument('--n_estimators', type=int, default=100, 
                        help='Number of estimators (trees) in LightGBM model')
    parser.add_argument('--output_path', type=str, default='models',
                        help='Path to save model and metrics')

    # Parse arguments
    args = parser.parse_args()

    # Train the model
    train_model(args.data_path, args.n_estimators, args.output_path)