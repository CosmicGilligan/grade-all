import requests
import json

# Your Canvas API URL and token
API_URL = 'https://sdccd.instructure.com/api/v1'
API_TOKEN = '1069~JHu89fDn3RfmMLCcP87FfXENWKMfkk3B9Y4hK7PaNQf3RGtrvDMPLCECeYZZQ846'
#course_id = '71354'  # Replace with the actual course ID

# Headers for the request
headers = {
    'Authorization': f'Bearer {API_TOKEN}'
}



# URL for listing all courses with additional parameters
url = f'{API_URL}/courses/'

params = {
    'per_page': 10000  # Setting the item count to 10 per page
}

# Make the GET request to retrieve courses
response = requests.get(url, headers=headers)
results = []
while url and len(results) < 20:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            raise Exception(f"API request failed with status code {response.status_code}")
        results.extend(response.json())
        # Check if there is a next page
        url = response.links.get('next', {}).get('url')
#return results[:max_items]

if response.status_code == 200:
    response_data = response.json()
#    for x in response_data:
#        if x[]

    # Specify the file path
    file_path = './response.json'

    # Save the data to a JSON file
    with open(file_path, 'w') as json_file:
        json.dump(response_data, json_file, indent=4)
        json_file.close()

    print(f"Data has been saved to {file_path}")
    with open(file_path, 'r') as json_file:
        print("File opened successfully")
        indata = json.load(json_file)
#        explore_json(indata)
        json_file.close()
else:
    print(f"Failed to retrieve data: {response.status_code}")

#explore_json(indata)


def explore_json(data, indent=0):
    spaces = ' ' * indent
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{spaces}{key}:")
            explore_json(value, indent + 2)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            print(f"{spaces}[{i}]:")
            explore_json(item, indent + 2)
    else:
        print(f"{spaces}{data}")

