import requests

url = "https://online-shoppers-intentions.onrender.com/predict"

shopper = {
    "administrative": 3,
    "administrative_duration": 87.5,
    "informational": 0,
    "informational_duration": 0.0,
    "productrelated": 27,
    "productrelated_duration": 798.33,
    "bouncerates": 0.0,
    "exitrates": 0.012,
    "pagevalues": 22.9,
    "specialday": 0.8,
    "month": "May",
    "operatingsystems": "2",
    "browser": "2",
    "region": "3",
    "traffictype": "1",
    "visitortype": "Returning_Visitor",
    "weekend": 0,
}

print(f"Sending POST request to cloud URL: ")
print(f"    {url}")
response = requests.post(url, json=shopper)

print(f"HTTP Status Code: ")
print(f"    {response.status_code}")
print(f"Response Server Header: ")
print(f"    {response.headers.get('Server', 'Unknown')}")
print("JSON Response:")
res_json = response.json()
print(res_json)

prob = res_json["purchase_probability"]

if prob >= 0.5:
    print("High intent shopper -> No discount needed, they are likely to buy anyway!")
elif prob >= 0.25:
    print("Hesitant shopper -> Sending 10% promotional discount to nudge them to buy!")
else:
    print("Low intent window-shopper -> Do not send discount.")
