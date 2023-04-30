import json


def read_secret(secret: str) -> str:
    with open("config/secrets.json", "r") as f:
        content = json.loads(f.read())

        return content[secret]
