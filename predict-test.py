import requests

url = "http://localhost:9696/predict"

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

response = requests.post(url, json=shopper).json()
print(response)

prob = response["purchase_probability"]

if prob >= 0.5:
    print("High intent shopper -> No discount needed, they are likely to buy anyway!")
elif prob >= 0.25:
    print("Hesitant shopper -> Sending 10% promotional discount to nudge them to buy!")
else:
    print("Low intent window-shopper -> Do not send discount.")
