import requests

# Define the API endpoint
url = "http://127.0.0.1:8000"

# Define the user input (replace with different questions to test)
data = {"text": "How do I reset my Dalhousie password?"}

# Send a POST request to the chatbot API
response = requests.post(url, json=data)

# Print the chatbot's response
print("Chatbot Response:", response.json())
