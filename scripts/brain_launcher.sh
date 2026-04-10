#!/bin/bash
# Observer Brain — Llama 3.2 1B Instruct (Non-thinking, fast, ARM64 optimized)
pkill -f "llama-server" 2>/dev/null; sleep 1
nohup /home/ubuntu/llama-cpp-source/build/bin/llama-server \
  -m /home/ubuntu/llama-cpp-source/models/llama3.2-1b-instruct-q8.gguf \
  --port 1234 --host 0.0.0.0 \
  --ctx-size 4096 --n-predict 512 \
  --alias "observer" -t 2 \
> /home/ubuntu/observer_brain.log 2>&1 &
echo "Observer Brain started (PID: $!)"
