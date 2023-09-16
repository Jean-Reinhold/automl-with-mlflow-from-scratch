#!/bin/bash

if ! command -v python3 &> /dev/null; then
    echo "python3 could not be found"
    exit
fi

if [ ! -d "venv" ]; then
  python3 -m venv venv
fi

source venv/bin/activate

if ! pip install -r requirements.txt; then
    echo "Failed to install requirements"
    deactivate
    exit
fi

if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null ; then
    echo "Port 5000 is in use, please close the process using it and try again."
    deactivate
    exit
fi

if ! pgrep -f "mlflow server" > /dev/null; then
    export MLFLOW_TRACKING_URI=http://0.0.0.0:5000
    
    mlflow server \
        --backend-store-uri sqlite:///mlflow/mlflow.db \
        --default-artifact-root ./mlflow/mlruns \
        --host 0.0.0.0 &
    sleep 10
fi

# Run application
if ! python main.py; then
    echo "Failed to run application"
fi

deactivate
