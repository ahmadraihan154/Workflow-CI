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

# Load Data
data_path = r'D:\6. Membangun Sistem Machine Learning\Workflow-CI\MLProject\diamond_preprocessing'
transformer = joblib.load(os.path.join(data_path, 'power_transformers.joblib'))
price_transformer = transformer['price']

X_train = pd.read_csv(os.path.join(data_path, 'X_train.csv'))
y_train = pd.read_csv(os.path.join(data_path, 'y_train.csv'))
X_test = pd.read_csv(os.path.join(data_path, 'X_test.csv'))
y_test = pd.read_csv(os.path.join(data_path, 'y_test.csv'))
y_test = price_transformer.inverse_transform(y_test.to_numpy().reshape(-1,1))

def train_model(n_estimators):
    mlflow.set_experiment('base-model_experiment_2')

    with mlflow.start_run(run_name='LGBM_Base_2'):
        model = LGBMRegressor(n_estimators=n_estimators)
        model.fit(X_train, y_train)  

        y_pred_transform = model.predict(X_test)
        y_pred = price_transformer.inverse_transform(y_pred_transform.reshape(-1,1))
        r2_skor = r2_score(y_test, y_pred)
        rmse_skor = np.sqrt(mean_squared_error(y_test, y_pred))

        mlflow.log_params(model.get_params())
        mlflow.log_metric("RMSE", rmse_skor)    
        mlflow.log_metric("R2", r2_skor)
        mlflow.sklearn.log_model(model, artifact_path="model")

    print(f'R2 Score : {r2_skor}')
    print(f'RMSE : {rmse_skor}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', type=int, default=100, help='Jumlah Pohon dalam Model LightGBM')
    args = parser.parse_args()

    train_model(args.n_estimators)