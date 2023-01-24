import pymongo.collection


def check_categories(
    categories_collection: pymongo.collection.Collection, category_name: str
):
    # If category name is changed category won't be found and new one will be created
    return categories_collection.find_one({"category_name": {"$eq": category_name}})


def add_categories(
    categories_collection: pymongo.collection.Collection,
    category_id: str,
    category_name: str,
    category_url: str,
):
    # Check first category is not in collection
    is_category_existent = check_categories(categories_collection, category_name)
    if not is_category_existent:
        # We suppose url exists
        new_category = {
            "category_id": category_id,
            "category_name": category_name,
            "category_url": category_url,
        }
        response = categories_collection.insert_one(new_category)

        if response:
            return True
    else:
        if (
            is_category_existent["category_id"] != category_id
            or is_category_existent["category_name"] != category_name
            or is_category_existent["category_url"] != category_url
        ):
            return update_categories(
                categories_collection=categories_collection,
                category_id=category_id,
                category_name=category_name,
                category_url=category_url,
            )
    return False


def update_categories(
    categories_collection: pymongo.collection.Collection,
    category_id: str,
    category_name: str,
    category_url: str,
):
    # Check if category exists
    category_to_update = check_categories(categories_collection, category_name)
    if category_to_update:
        # We suppose the url is correct
        response = categories_collection.update_one(
            {"category_name": category_name},
            {
                "$set": {
                    "category_id": category_id,
                    "category_name": category_name,
                    "category_url": category_url,
                }
            },
        )
        if response:
            return True
    return False
