from app.db_conn import get_categories_collection, get_products_collection
from db.db_connection import get_db_connection
from db.add_products import add_products_from_json
from db.create_update_categories import add_categories, update_categories

categories_collection = get_categories_collection()

# result = add_categories(
#     categories_collection=categories_collection,
#     category_id=str(uuid.uuid4()),
#     category_name="Computers",
#     category_url="https://www.amazon.es/gp/bestsellers/computers/ref=zg_bs_nav_0",
# )

# result = update_categories(
#     categories_collection=categories_collection,
#     category_name="Guitarras y accesorios",
#     category_url="https://www.amazon.es/gp/bestsellers/musical-instruments/4965355031/ref=zg_bs_nav_musical-instruments_1",
# )

result = add_products_from_json(
    product_collection=get_products_collection(),
    categories_collection=get_categories_collection(),
    json_file_path="db/json_products/cars_bestsellers.json",
)


print(result)
