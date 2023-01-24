import pymongo.collection
import uuid

from .db_schema import users_schema

def check_user(users_collection: pymongo.collection.Collection, username: str):
    return users_collection.find_one({"username": {"$eq": username}})

def add_user_to_db(users_collection: pymongo.collection.Collection, user_to_add: dict) -> int:
    """
    return types:
        0 -> User added
        1 -> User already existing
    """
    # First check if user already exists
    if not check_user(users_collection=users_collection, username=user_to_add['username']):
        user_with_id = {"id": str(uuid.uuid4())}
        user_with_id.update(user_to_add)

        users_schema.validate(user_with_id)
        users_collection.insert_one(user_with_id)

        return 0
    else:
        return 1
