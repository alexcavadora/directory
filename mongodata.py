# import pandas as pd
# from pymongo import MongoClient

# df = pd.read_csv('data/clean.csv', delimiter=';')

# client = MongoClient('mongodb://alex:Z@localhost:27017/?authSource=admin')
# db = client['directory']
# collection = db['lidia']

# for index, row in df.iterrows():
#     document = {
#         "description": row['description'],
#         "name": row['name']
#     }
#     collection.insert_one(document)

# print("Data inserted successfully!")

# for doc in collection.find():
#     print(doc)
import json
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://alex:Z@localhost:27017/?authSource=admin')
db = client['directory']

# Function to load JSON data and insert into a specified MongoDB collection
def load_json_to_mongo(file_path, collection_name):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        
        # Check if the data is a dictionary with the appropriate key
        if 'sinonimos' in data:
            items = data['sinonimos']
        elif 'antonimos' in data:
            items = data['antonimos']
        else:
            print(f"Unexpected JSON structure in {file_path}")
            return
        
        # Insert each item into the collection
        for key, values in items.items():
            document = {
                "palabra": key,
                "sinonimos" if 'sinonimos' in data else "antonimos": values
            }
            try:
                db[collection_name].insert_one(document)
            except Exception as e:
                print(f"Error inserting {key}: {e}")

# Load sinonimos.json
load_json_to_mongo('data/sinonimos.json', 'sinonimos')

# Load antonimos.json
load_json_to_mongo('data/antonimos.json', 'antonimos')

print("Sinónimos and Antónimos inserted successfully!")