# run.sh
#!/bin/bash

docker run --gpus all \
    -e HYDRA_FULL_ERROR=1 \
    -v /mnt/ssd2:/mnt/ssd2:ro \
    -v /mnt/ssd3:/mnt/ssd3:ro \
    -v $(pwd)/outputs:/app/outputs \
    -v $(pwd)/mlflow.db:/app/mlflow.db \
    -v $(pwd)/checkpoints:/app/checkpoints \
    -v $(pwd)/main.py:/app/main.py \
    -v $(pwd)/src:/app/src \
    -v $(pwd)/configs:/app/configs \
    microplastics-predict \
    python main.py "$@"
