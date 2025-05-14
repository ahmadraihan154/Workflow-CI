import mlflow
import pandas as pd
from lightgbm import LGBMRegressor
import os
import numpy as np
import warnings
import sys
import joblib
from sklearn.metrics import mean_squared_error, r2_score 

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the preprocessed data from CSV (set default if not provided)
    file_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "diamond_preprocessing")
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'models'

    X_train = pd.read_csv(os.path.join(file_path, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(file_path, 'y_train.csv'))
    X_test = pd.read_csv(os.path.join(file_path, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(file_path, 'y_test.csv'))

    # Load the transformer (assuming you have saved the transformer during preprocessing)
    transformer = joblib.load(os.path.join(file_path, 'power_transformers.joblib'))
    price_transformer = transformer['price']  # Assuming price transformer is included

    # Inverse transform the y_test to original scale (before transformation)
    y_test = price_transformer.inverse_transform(y_test.to_numpy().reshape(-1, 1))

    # Example input for MLflow logging
    input_example = X_train[0:5]

    # Start MLflow run
    with mlflow.start_run():
        # Define and train the model
        model = LGBMRegressor()
        model.fit(X_train, y_train)

        # Predictions
        y_pred_transform = model.predict(X_test)

        # Inverse transform the predictions to original scale
        y_pred = price_transformer.inverse_transform(y_pred_transform.reshape(-1, 1))

        # Log the model to MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )

        # Log metrics
        r2_skor = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mlflow.log_metric("R2", r2_skor)
        mlflow.log_metric('RMSE', rmse)

        # Ensure the output directory exists before saving the model
        os.makedirs(output_path, exist_ok=True)

        # Save model and metrics for GitHub Actions or other output paths
        model_file = os.path.join(output_path, 'lgbm_model.joblib')
        joblib.dump(model, model_file)

        metrics = {'r2_score': r2_skor}
        metrics_file = os.path.join(output_path, 'metrics.joblib')
        joblib.dump(metrics, metrics_file)

        print(f"Model saved to {model_file}")
        print(f'RMSE : {rmse}')
        print(f"R2 Score: {r2_skor}")
