from db.db_connection import get_db_connection


def db_conn():
    return get_db_connection()


def get_users_collection():
    return db_conn()["users"]


def get_products_collection():
    return db_conn()["products"]


def get_categories_collection():
    return db_conn()["categories"]


def get_product_rating_collection():
    return db_conn()["product_rating"]
