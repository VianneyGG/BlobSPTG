import os
import requests

# Folder to save the files
target_folder = "tests"
os.makedirs(target_folder, exist_ok=True)

# Define the ranges for X and Y
x_range = ['b', 'c', 'd', 'e']
y_range = list(range(1, 21))

# For 'b', only go up to 18
y_limit = {'b': 18, 'c': 20, 'd': 20, 'e': 20}

base_url = "http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/stein{}{}.txt"

for x in x_range:
    for y in y_range:
        if y > y_limit[x]:
            continue
        url = base_url.format(x, y)
        filename = f"stein{x}{y}.txt"
        filepath = os.path.join(target_folder, filename)
        try:
            print(f"Downloading {url} ...")
            response = requests.get(url)
            response.raise_for_status()
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(response.text)
            print(f"Saved to {filepath}")
        except Exception as e:
            print(f"Failed to download {url}: {e}")

