#!/bin/bash

#MAIN_MODEL_NAME=meta-llama/Llama-3.2-3B-Instruct
MAIN_MODEL_NAME=hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
MAIN_MODEL_ESCAPED_NAME=$(echo $MAIN_MODEL_NAME | sed 's/\//-/g')
MAIN_MODEL_PATH=models_cache/$MAIN_MODEL_ESCAPED_NAME
#EMBED_MODEL_NAME=sentence-transformers/all-roberta-large-v1
EMBED_MODEL_NAME=intfloat/multilingual-e5-large
EMBED_MODEL_ESCAPED_NAME=$(echo $EMBED_MODEL_NAME | sed 's/\//-/g')
EMBED_MODEL_PATH=models_cache/$EMBED_MODEL_ESCAPED_NAME
#HF_TOKEN=...
FLAGFILE_INSTALL="venv/.completed_installation"
FLAGFILE_MAIN_MODEL=models_cache/$MAIN_MODEL_ESCAPED_NAME/.completed_download
FLAGFILE_EMBED_MODEL=models_cache/$EMBED_MODEL_ESCAPED_NAME/.completed_download
VLLM_LOGGING_LEVEL=DEBUG

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

if [ ! -f $FLAGFILE_INSTALL ]; then
    venv/bin/python3 -m pip install --upgrade pip
    venv/bin/python3 -m pip install -U -r requirements.txt
    touch $FLAGFILE_INSTALL
fi

if [ ! -d "models_cache" ]; then
    mkdir models_cache
fi

if [ ! -f $FLAGFILE_MAIN_MODEL ]; then
    HF_HUB_ENABLE_HF_TRANSFER=1 venv/bin/huggingface-cli download $MAIN_MODEL_NAME --local-dir ./models_cache/$MAIN_MODEL_ESCAPED_NAME --exclude "*.pth"
    touch $FLAGFILE_MAIN_MODEL
fi

if [ ! -f $FLAGFILE_EMBED_MODEL ]; then
    HF_HUB_ENABLE_HF_TRANSFER=1 venv/bin/huggingface-cli download $EMBED_MODEL_NAME --local-dir ./models_cache/$EMBED_MODEL_ESCAPED_NAME --exclude "*.onnx" "*.bin" "onnx/*"
    touch $FLAGFILE_EMBED_MODEL
fi

if ! command -v tmux &> /dev/null
then
    sudo apt-get install tmux
fi

if [ -z "$(tmux list-sessions | grep vllm_server_main)" ]; then
    echo "Starting vllm_server_main"
    tmux new-session -d -s vllm_server_main \
        'venv/bin/vllm serve '$MAIN_MODEL_PATH' \
        --task generate \
        --dtype half \
        --port 8999 \
        --gpu-memory-utilization 0.75 \
        --max-model-len 8192 \
        --host 0.0.0.0 && exit'
fi

if [ -z "$(tmux list-sessions | grep vllm_server_embedding)" ]; then
    echo "Starting vllm_server_embedding"
    tmux new-session -d -s vllm_server_embedding \
        'venv/bin/vllm serve '$EMBED_MODEL_PATH' \
        --task embed \
        --dtype half \
        --port 8998 \
        --gpu-memory-utilization 0.25 \
        --host 0.0.0.0 && exit'

#    --quantization gptq --dtype half \
#    --enable-auto-tool-choice \
#    --quantization gptq \
fi

while ! curl -s localhost:8998/v1/models | grep -q $EMBED_MODEL_ESCAPED_NAME; do
    echo "Waiting for vllm_server_embedding to start..."
    sleep 2
done

#while ! curl -s localhost:8999/v1/models | grep -q $MAIN_MODEL_ESCAPED_NAME; do
if false; then
    echo "Waiting for vllm_server_main to start..."
    sleep 2
#done
fi

if [ -z "$(tmux list-sessions | grep chatbot_app)" ]; then
    echo "Starting chatbot_app"
    tmux new-session -d -s chatbot_app \
        'venv/bin/python3 app.py && exit'
    #venv/bin/python3 app.py
fi