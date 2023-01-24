import json
import requests

params = {
    'api_key': 'FC72AC5BAB5D448BB0DD5FE83C40E95F',
    'type': 'bestsellers',
    'url': 'https://www.amazon.es/gp/bestsellers/musical-instruments/4965355031/ref=zg_bs_nav_musical-instruments_1',
    'output': 'json'
}

# make the http GET request to Rainforest API
api_result = requests.get('https://api.rainforestapi.com/request', params)

# print the JSON response from Rainforest API
json_object = json.dumps(api_result.json())

with open("output_guitars.json", "w") as outfile:
    outfile.write(json_object)