#!/bin/bash
# AI-CodeCompass Observer — Automated System Scanner
# File: /home/ubuntu/AI-CodeCompass/scripts/scanner.sh

# 1. Gather Activity (Files changed in the last 15 minutes)
ACTIVITY=$(find /home/ubuntu -maxdepth 2 -not -path "*/.*" -mmin -15)

if [ -z "$ACTIVITY" ]; then
    ACTIVITY="No significant changes detected in the last 15 minutes."
fi

# 2. Support for Tool Mode (Output raw data)
if [ "$1" == "--raw" ]; then
    echo "$ACTIVITY"
    exit 0
fi

# 3. Trigger the Autonomous Observer (with the new Super-Hero Brain)
echo "------------------------------------------------"
echo "Initializing AI-CodeCompass Observer Audit..."
/home/ubuntu/llm_pipeline_env/bin/python3 /home/ubuntu/AI-CodeCompass/observer.py --auto
echo "------------------------------------------------"
