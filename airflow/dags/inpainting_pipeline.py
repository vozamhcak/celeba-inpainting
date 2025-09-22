from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

def prepare_data():
    os.system("python code/prepare_data.py")

def train_model():
    os.system("python code/models/train.py")

def deploy_api():
    os.system("docker compose up -d --build")

default_args = {
    'owner': 'mlops-student',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'retries': 1,
}

with DAG(
    dag_id='inpainting_pipeline',
    default_args=default_args,
    description='Face inpainting full pipeline',
    schedule_interval='@daily',
    catchup=False,
) as dag:

    task_prepare = PythonOperator(
        task_id='prepare_data',
        python_callable=prepare_data,
    )

    task_train = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
    )

    task_deploy = PythonOperator(
        task_id='deploy_api',
        python_callable=deploy_api,
    )

    task_prepare >> task_train >> task_deploy
