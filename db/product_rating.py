import pymongo.collection

from .db_schema import rating_schema
from .add_products import is_product_in_db_by_asin


def set_product_rating_in_db(
    product_rating_collection: pymongo.collection.Collection,
    rating: dict,
):
    rating_schema.validate(rating)
    product_rating_collection.insert_one(rating)


def delete_product_rating_in_db(
    product_rating_collection: pymongo.collection.Collection,
):
    product_rating_collection.delete_many({})
