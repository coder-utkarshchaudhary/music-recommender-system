import json

with open(r'API hits\analysis.json', 'r') as file:
    raw_data = json.load(file)

print(raw_data[0].keys())

with open(r'API hits\features.json', 'r') as file:
    raw_data = json.load(file)

print(raw_data[0].keys())