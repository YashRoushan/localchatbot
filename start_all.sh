#!/bin/bash

echo "Starting Ollama server..."
ollama serve &  # Start Ollama in the background
sleep 2  # Wait for Ollama to initialize

echo "Starting Chatbot..."
cd ~/Desktop/localchatbot
uvicorn chatbot:app --host 0.0.0.0 --port 8000 --reload

echo "Chatbot started successfully!"
