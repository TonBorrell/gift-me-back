from fastapi import APIRouter

import os
from dotenv import load_dotenv

from db.add_products import get_products
from ..db_conn import get_products_collection

router = APIRouter(
    prefix="/products", tags=["products"], responses={404: {"description": "Not found"}}
)


@router.get("/")
async def get_products_api() -> list:
    products = get_products(get_products_collection())
    response = []
    for prod in products:
        prod.pop("_id", None)
        response.append(prod)

    return response
