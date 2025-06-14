from dotenv import load_dotenv
import os
from pymongo.mongo_client import MongoClient
import pandas as pd

load_dotenv()

MONGO_DB_URL = os.getenv('MONGO_DB_URL')
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME")

print(f"DB_URL: {MONGO_DB_URL}")
print(f"DB_NAME: {MONGO_DB_NAME}")
print(f"COLLECTION_NAME: {MONGO_COLLECTION_NAME}")


# Create a new client and connect to the server
client = MongoClient(MONGO_DB_URL)
db = client[MONGO_DB_NAME]
collection = db[MONGO_COLLECTION_NAME]

# Fetch all documents
documents = list(collection.find())

# Convert to DataFrame
df = pd.DataFrame(documents)

# Drop the MongoDB _id column if present
if "_id" in df.columns:
    df = df.drop(columns=["_id"])

# Print shape and preview
print("Shape of DataFrame:", df.shape)
print(df.head())

# try:
#     client.admin.command('ping')
#     print("Pinged your deployment. You successfully connected to MongoDB!")
# except Exception as e:
#     print(e)