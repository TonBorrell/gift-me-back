import json


def get_dic_from_json(file_path: str) -> dict:
    with open(file_path, "r") as outfile:
        json_info = json.load(outfile)

    products = {}

    for product in json_info["bestsellers"]:
        product_ind = {
            "asin": product["asin"],
            "name": product["title"],
            "link": product["link"],
            "image": product["image"],
            "price": float(product["price"]["value"]),
            "category": {
                "name": product["current_category"]["name"],
                "category_id": product["current_category"]["id"],
                "category_url": product["current_category"]["link"],
            },
        }

        rank = product["rank"]
        id_category = product["current_category"]["name"]

        products[f"{rank}_{id_category}"] = product_ind

    return products
