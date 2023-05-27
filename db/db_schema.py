from schema import Schema, And, Use

product_schema = Schema(
    {
        "id": And(str, len),
        "asin": And(str, len),
        "name": And(str, len),
        "link": And(str, len),
        "image": And(str, len),
        "price": float,
        "category": dict,
    }
)

users_schema = Schema(
    {
        "id": And(str, len),
        "email": And(str, len),
        "username": And(str, len),
        "password": And(str, len),
        "timestamp": And(str, len),
    }
)

rating_schema = Schema(
    {"name": str, "age": str, "preferences": dict[str, bool], "rating": dict[str, int]}
    # {"asin": And(str, len), "rating": And(Use(int), lambda n: 0 <= n <= 5)}
)
