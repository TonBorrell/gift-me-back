from typing import Dict
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import random

from db.add_products import get_products
from db.product_rating import set_product_rating_in_db
from ..db_conn import get_products_collection, get_product_rating_collection

router = APIRouter(
    prefix='/product_rating',
    tags=['product_rating'],
    responses={404: {"description": "Not found"}}
)

class Rating(BaseModel):
    asin: str
    rating: int

@router.get('/')
def get_products_to_rate() -> list:
    products = get_products(get_products_collection())
    response = []
    for prod in products:
        prod.pop('_id', None)
        response.append(prod)
    
    random_response = random.choices(response, k=10)

    return random_response

@router.get('/one')
def get_products_to_rate() -> list:
    products = get_products(get_products_collection())
    response = []
    for prod in products:
        prod.pop('_id', None)
        response.append(prod)
    
    random_response = random.choices(response, k=1)

    return random_response

@router.post('/')
def set_product_rating(rating: Dict[str, str]):
    print(rating)
    for asin, rate in rating.items():
        print(asin, rate)
    return {'success': 'all rating wrote on db'}
    """
    for prod_rating in rating:
        set_product_rating_in_db(product_collection=get_products_collection(), product_rating_collection=get_product_rating_collection(), rating=prod_rating.dict())
    return {'success': 'all rating wrote on db'}
    """

