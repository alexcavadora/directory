import pandas as pd
from pymongo import MongoClient

# Load the CSV data
df = pd.read_csv('data/clean.csv', delimiter=';')

# Connect to MongoDB
client = MongoClient('mongodb://alex:Z@localhost:27017/?authSource=admin')
db = client['directory']
collection = db['lidia']

# Insert data into MongoDB
for index, row in df.iterrows():
    document = {
        "description": row['description'],
        "name": row['name']
    }
    collection.insert_one(document)

print("Data inserted successfully!")

for doc in collection.find():
    print(doc)
