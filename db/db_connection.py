import pymongo

def get_db_connection():
    client = pymongo.MongoClient("mongodb+srv://admin:admin@giftme.rgncsqt.mongodb.net/test")
    db = client['gift-me']

    return db