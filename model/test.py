import requests

url = "http://localhost:9696/predict"

# order must match training exactly
car_data = {
    "displacement": 307.0,
    "cylinders": 8,
    "horsepower": 130.0,
    "weight": 3504.0,
    "acceleration": 12.0,
    "model_year": 70,
    "origin": 1
}

response = requests.post(url, json=car_data)

print("Status Code:", response.status_code)
try:
    print("Response JSON:", response.json())
except Exception as e:
    print("Raw Response Text:", response.text)
