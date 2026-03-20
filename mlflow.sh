# mlflow.sh
#!/bin/bash

docker run --rm \
    -p 5000:5000 \
    -v $(pwd)/mlflow.db:/app/mlflow.db \
    microplastics-predict \
    mlflow ui \
        --backend-store-uri sqlite:////app/mlflow.db \
        --host 0.0.0.0 \
        --port 5000
