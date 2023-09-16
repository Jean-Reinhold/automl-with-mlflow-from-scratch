#!/bin/bash

if [ ! -d "venv" ]; then
  python3 -m venv venv
fi

source venv/bin/activate

pip install -r requirements.txt

if ! pgrep -f "mlflow server" > /dev/null; then
    export MLFLOW_TRACKING_URI=http://0.0.0.0:5000
    
    mlflow server \
        --backend-store-uri sqlite:///mlflow.db \
        --default-artifact-root ./mlruns \
        --host 0.0.0.0 &
    sleep 10
fi

python main.py 