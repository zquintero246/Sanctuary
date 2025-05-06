#!/bin/bash

echo "🧹 Limpiando GPU..."

nvidia-smi

sudo pkill -f python
sudo pkill -f ollama


sleep 1
echo "✅ GPU lista."

nvidia-smi

