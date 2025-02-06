#!/bin/bash

# Navigate to the chatbot directory
cd ~/Desktop/localchatbot

# Activate Python environment (only if using a virtualenv)
# source ~/your_virtualenv/bin/activate

# Start FastAPI chatbot
uvicorn chatbot:app --host 0.0.0.0 --port 8000 --reload &

echo "Chatbot started successfully!"
