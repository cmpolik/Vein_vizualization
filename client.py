import requests

response = requests.get('http://192.168.1.2:5000/data')
print(response.json())

