from fastapi import FastAPI, HTTPException
from loguru import logger

app = FastAPI()

users: dict[str, str] = {}

logger.add("logs/users.log", format="{level} {time} {message}", level="INFO")


@app.get("/login")
def login_get() -> dict[str, str]:
    return {"status": "success"}


@app.post("/login")
def login(username: str, password: str) -> dict[str, str]:
    if username not in users.keys():
        users[username] = password

        logger.info(f"Created user: {username}")

        return {"status": "success"}
    raise HTTPException(status_code=404, detail="Item not found")
