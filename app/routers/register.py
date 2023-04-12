from datetime import datetime
from fastapi import APIRouter, HTTPException, Request
from passlib.context import CryptContext
from pydantic import BaseModel

from ..db_conn import get_users_collection
from db.manage_users import add_user_to_db

router = APIRouter(
    prefix='/register',
    tags=['register'],
    responses={404: {"description": "Not found"}}
)

class User(BaseModel):
    email: str
    username: str
    password: str
    timestamp: str

def encrypt_password(password: str) -> str:
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    return pwd_context.verify(plain_password, hashed_password)

@router.post('/')
async def register_user(user: User):
    user.password = encrypt_password(password=user.password)
    response = add_user_to_db(users_collection=get_users_collection(), user_to_add=user.dict())
    if response == 0:
        return {"status": "User created"}
    elif response == 1:
        return {"status": "User already existing"}
    raise HTTPException(status_code=404, detail="User not created")

