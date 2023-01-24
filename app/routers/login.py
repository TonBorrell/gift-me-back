from datetime import datetime
from fastapi import APIRouter, HTTPException, Request
from passlib.context import CryptContext
from pydantic import BaseModel

from ..db_conn import get_users_collection
from db.manage_users import check_user

router = APIRouter(
    prefix='/login',
    tags=['login'],
    responses={404: {"description": "Not found"}}
)

class User(BaseModel):
    username: str
    password: str

def verify_password(plain_password, hashed_password):
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    return pwd_context.verify(plain_password, hashed_password)

@router.post('')
async def register_user(user: User):
    user_db = check_user(users_collection=get_users_collection(), username=user.username)
    if user_db:
        is_password_correct = verify_password(user.password, user_db['password'])
        if is_password_correct:
            return {"Logged in": True}
        else:
            return {"Logged in": False}
    elif user_db == None:
        return {"Logged in": False}
    raise HTTPException(status_code=404, detail="User not found")

