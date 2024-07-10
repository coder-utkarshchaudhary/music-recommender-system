import base64
import requests as r
import json

client_id = '3415b4865e5b4f0aacb586f52c95332d'
client_secret = '8eb10cc35ac64d0aba7322f123df252b'

credentials = f"{client_id}:{client_secret}"
base64_credentials = base64.b64encode(credentials.encode()).decode()

# Get the access token
url = "https://accounts.spotify.com/api/token"
headers = {
    "Authorization": f"Basic {base64_credentials}",
    "Content-Type": "application/x-www-form-urlencoded"
}
data = {
    "grant_type": "client_credentials"
}
response = r.post(url, headers=headers, data=data)
access_token = response.json().get("access_token")

feature_url = "https://api.spotify.com/v1/audio-features/"
analysis_url = "https://api.spotify.com/v1/audio-analysis/"

id = "11dFghVXANMlKmJXsNCbNl"

headers = {
    "Authorization": f"Bearer {access_token}"
}

features = r.get(url=feature_url+id, headers=headers)
analysis = r.get(url=analysis_url+id, headers=headers)

if features.status_code == 200:
    new_data = features.json()
 
    existing_data = []
 
    existing_data.append(new_data)
 
    with open(r"API hits\features.json", "w") as json_file:
        json.dump(existing_data, json_file, indent=4)
        print("Data appended to feature.json file.")
else:
    print("Failed to retrieve data from the API. Status code:", features.status_code)

if analysis.status_code == 200:
    new_data = analysis.json()
 
    existing_data = []
 
    existing_data.append(new_data)
 
    with open(r"API hits\analysis.json", "w") as json_file:
        json.dump(existing_data, json_file, indent=4)
        print("Data appended to analysis.json file.")
else:
    print("Failed to retrieve data from the API. Status code:", analysis.status_code)