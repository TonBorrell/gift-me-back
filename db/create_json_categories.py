import json
import requests

from functions.read_secrets import read_secret


rainforest_api_key = read_secret("RAINFOREST_API_KEY")

categories_url = {
    "cars": "https://www.amazon.es/gp/bestsellers/automotive/ref=zg_bs_nav_0",
    "guitars": "https://www.amazon.es/gp/bestsellers/musical-instruments/4965355031/ref=zg_bs_nav_musical-instruments_1",
    "electronics": "https://www.amazon.es/gp/bestsellers/electronics/ref=zg_bs_nav_0",
    "computers": "https://www.amazon.es/gp/bestsellers/computers/ref=zg_bs_nav_0",
    "kitchen": "https://www.amazon.es/gp/bestsellers/kitchen/ref=zg_bs_nav_0",
}

for category, url in categories_url.items():
    params = {
        "api_key": rainforest_api_key,
        "type": "bestsellers",
        "url": url,
        "output": "json",
    }

    # make the http GET request to Rainforest API
    api_result = requests.get("https://api.rainforestapi.com/request", params)

    # print the JSON response from Rainforest API
    json_object = json.dumps(api_result.json())

    print(json_object)

    with open(f"db/json_products/{category}_bestsellers.json", "w") as outfile:
        outfile.write(json_object)
