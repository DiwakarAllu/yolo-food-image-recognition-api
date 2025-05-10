import requests

url = "http://localhost:5000/yolo_predict"
files = {"image": open("p.jpg", "rb")}
response = requests.post(url, files=files)

print(response.status_code)
print(response.json())
