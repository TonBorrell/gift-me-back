from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .routers import products, register, login, product_rating, model_product

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(products.router)
app.include_router(register.router)
app.include_router(login.router)
app.include_router(product_rating.router)
app.include_router(model_product.router)
