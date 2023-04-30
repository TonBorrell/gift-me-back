import pymongo

from functions.read_secrets import read_secret


def get_db_connection():
    mongo_client = read_secret("MONGO_CLIENT")
    client = pymongo.MongoClient(mongo_client)
    db = client["gift-me"]

    return db
