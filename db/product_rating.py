import pymongo.collection

from .db_schema import rating_schema
from .add_products import is_product_in_db_by_id 

def set_product_rating_in_db(product_collection: pymongo.collection.Collection, product_rating_collection: pymongo.collection.Collection, rating: dict):
    rating_schema.validate(rating)
    if is_product_in_db_by_id(product_collection, rating['product_id']):
        product_rating_collection.insert_one(rating)
