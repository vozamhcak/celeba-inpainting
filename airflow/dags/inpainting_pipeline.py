from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import subprocess
import logging
import os


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

def run_script(command, cwd=None):
    logging.info(f"Running command: {command}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd or BASE_DIR,
            capture_output=True,
            text=True,
        )
        logging.info(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            logging.error(f"STDERR:\n{result.stderr}")
        result.check_returncode()
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with return code {e.returncode}")
        raise

def prepare_data():
    script_path = os.path.join(BASE_DIR, "code", "prepare_data.py")
    run_script(f"python3 {script_path}")

def train_model():
    script_path = os.path.join(BASE_DIR, "code", "models", "train.py")
    run_script(f"python3 {script_path}")

def deploy_api():
    compose_path = os.path.join(BASE_DIR, "docker-compose.yml")
    run_script(f"docker compose -f {compose_path} up -d --build", cwd=BASE_DIR)

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
