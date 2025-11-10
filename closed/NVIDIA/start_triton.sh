#!/bin/bash
MODEL_REPO=/path/to/repo_0 #created by triton config builds
TRITON_BIN=/opt/tritonserver/bin/tritonserver
GRPC_PORT=8001
HTTP_PORT=8000
METRICS_PORT=8002

WORLD_SIZE= #number of GPUs per node

exec mpirun -n ${WORLD_SIZE} --allow-run-as-root \
  ${TRITON_BIN} \
    --model-repository=${MODEL_REPO} \
    --grpc-port=${GRPC_PORT} \
    --http-port=${HTTP_PORT} \
    --metrics-port=${METRICS_PORT}
