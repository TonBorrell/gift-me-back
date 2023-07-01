from typing import Dict
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import os
from joblib import load
import pandas as pd
import json

from db.add_products import get_products, is_product_in_db_by_asin
from db.product_rating import set_product_rating_in_db, delete_product_rating_in_db
from ..db_conn import get_products_collection, get_product_rating_collection

router = APIRouter(
    prefix="/api/v1/model_product",
    tags=["model_product"],
    responses={404: {"description": "Not found"}},
)


class Preferences_post(BaseModel):
    name: str
    age: str
    preferences: dict[str, bool]


def create_df_to_predict(
    preferences: Preferences_post,
    products,
    asin_dict: dict[str, int],
    cat_dict: dict[str, int],
):
    dataframe = []
    for document in products:
        if document["asin"] in asin_dict:
            actual_doc = {
                "age": preferences.age,
                "coffee": preferences.preferences["coffee"],
                "cooking": preferences.preferences["cooking"],
                "sports": preferences.preferences["sports"],
                "cars": preferences.preferences["cars"],
                "technology": preferences.preferences["technology"],
                "garden": preferences.preferences["garden"],
                "asin": asin_dict[document["asin"]],
                "category": cat_dict[document["category"]["category_id"]],
                "price": document["price"],
            }
            dataframe.append(actual_doc)

    return pd.DataFrame.from_dict(dataframe)


@router.post("/")
def set_product_rating(preferences: Preferences_post):
    # TODO: Carregar model amb dades preferencies i retornar producte de prediccio
    products = get_products(get_products_collection())
    dir_path = os.path.dirname(os.path.realpath(""))
    with open(f"{dir_path}/gift-me-back/ml/model_data/asin_dict.json", "r") as file:
        asin_dict = json.loads(file.read())
    with open(
        f"{dir_path}/gift-me-back/ml/model_data/categories_dict.json", "r"
    ) as file:
        categories_dict = json.loads(file.read())

    model = load(f"{dir_path}/gift-me-back/ml/model_res/KNN.joblib")

    df = create_df_to_predict(preferences, products, asin_dict, categories_dict)
    predictions = model.predict(df)

    index_max = predictions.argmax()
    asin_max = df.iloc[index_max]["asin"]
    asin = list(asin_dict.keys())[list(asin_dict.values()).index(asin_max)]
    product_recommended = is_product_in_db_by_asin(get_products_collection(), asin)
    product_recommended.pop("_id", None)
    return product_recommended
