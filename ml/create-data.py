import pandas as pd
import pymongo
import json


def read_secret(secret: str) -> str:
    with open("config/secrets.json", "r") as f:
        content = json.loads(f.read())

        return content[secret]


def get_db_connection():
    mongo_client = read_secret("MONGO_CLIENT")
    client = pymongo.MongoClient(mongo_client)

    db = client["gift-me"]

    return db


if __name__ == "__main__":
    db = get_db_connection()
    product_rating_collection = db["product_rating"]
    cursor = product_rating_collection.find({})
    products_collection = db["products"]
    categories = {}
    asin_dict = {}
    cont_cat = 0
    cont_asin = 0
    cont_document = 0

    dataframe = []
    dataframe_no_cat = []
    for document in cursor:
        print(cont_document)
        cont_document += 1
        for asin, rating in document["rating"].items():
            actual_doc = {}
            actual_doc_no_cat = {}
            actual_doc["age"] = int(document["age"])
            actual_doc["coffee"] = document["preferences"]["coffee"]
            actual_doc["cooking"] = document["preferences"]["cooking"]
            actual_doc["sports"] = document["preferences"]["sports"]
            actual_doc["cars"] = document["preferences"]["cars"]
            actual_doc["technology"] = document["preferences"]["technology"]
            actual_doc["garden"] = document["preferences"]["garden"]
            if asin not in asin_dict.keys():
                asin_dict[asin] = cont_asin
                cont_asin += 1
            actual_doc["asin"] = asin_dict[asin]
            product = products_collection.find({"asin": asin})
            category = product[0]["category"]["category_id"]
            if category not in categories.keys():
                categories[category] = cont_cat
                cont_cat += 1
            actual_doc["category"] = categories[category]
            actual_doc["price"] = product[0]["price"]
            actual_doc["rating"] = rating
            dataframe.append(actual_doc)
            actual_doc_no_cat["age"] = int(document["age"])
            actual_doc_no_cat["coffee"] = document["preferences"]["coffee"]
            actual_doc_no_cat["cooking"] = document["preferences"]["cooking"]
            actual_doc_no_cat["sports"] = document["preferences"]["sports"]
            actual_doc_no_cat["cars"] = document["preferences"]["cars"]
            actual_doc_no_cat["technology"] = document["preferences"]["technology"]
            actual_doc_no_cat["garden"] = document["preferences"]["garden"]
            actual_doc_no_cat["rating"] = rating
            dataframe_no_cat.append(actual_doc_no_cat)

    print(dataframe)

    df = pd.DataFrame.from_dict(dataframe)
    df_no_cat = pd.DataFrame.from_dict(dataframe_no_cat)

    df.to_parquet("ml/model_data/full_dataframe.parquet")
    df.to_parquet("ml/model_data/full_dataframe_no_categories.parquet")

    with open("ml/model_data/asin_dict.json", "w") as outfile:
        json.dump(asin_dict, outfile)

    with open("ml/model_data/categories_dict.json", "w") as outfile:
        json.dump(categories, outfile)

    X = df.loc[:, df.columns != "rating"]
    y = df.loc[:, df.columns == "rating"]
    X_nocat = df_no_cat.loc[:, df_no_cat.columns != "rating"]
    y_nocat = df_no_cat.loc[:, df_no_cat.columns == "rating"]
