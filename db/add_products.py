import pymongo.collection
from .read_json import get_dic_from_json
from .db_schema import product_schema
from .create_update_categories import add_categories
import uuid


def is_product_in_db_by_asin(collection: pymongo.collection.Collection, asin: str):
    return collection.find_one({"asin": {"$eq": asin}})

def is_product_in_db_by_id(collection: pymongo.collection.Collection, id: str):
    return collection.find_one({"id": {"$eq": id}})

def get_products(collection: pymongo.collection.Collection):
    return collection.find()

def add_products_from_json(
    product_collection: pymongo.collection.Collection,
    categories_collection: pymongo.collection.Collection,
    json_file_path: str,
):
    prod_dic: dict = get_dic_from_json(json_file_path)

    for _, product_info in prod_dic.items():
        if not is_product_in_db_by_asin(product_collection, product_info["asin"]):
            # Check if category exists in categories collection
            add_categories(
                categories_collection=categories_collection,
                category_id=product_info["category"]["category_id"],
                category_name=product_info["category"]["name"],
                category_url=product_info["category"]["category_url"],
            )
            product_info_with_id = {"id": str(uuid.uuid4())}
            product_info_with_id.update(product_info)

            product_schema.validate(product_info_with_id)
            product_collection.insert_one(product_info_with_id)

    return True
