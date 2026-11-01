import ast
import gzip
import json
from pathlib import Path

import pandas as pd


DATA = Path(__file__).resolve().parent / "data"
RAW = DATA / "raw"

def read_gzip(path):
    with gzip.open(path, "rt", encoding="utf-8") as file:
        for line in file:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield ast.literal_eval(line)


reviews = []

for item in read_gzip(
    RAW / "reviews_Cell_Phones_and_Accessories_5.json.gz"
):
    reviews.append({
        "user_id": item["reviewerID"],
        "product_id": item["asin"],
        "rating": item["overall"],
        "timestamp": item["unixReviewTime"]
    })

ratings = pd.DataFrame(reviews)
ratings = ratings.drop_duplicates(
    ["user_id", "product_id"], keep="last"
)

top_products = ratings["product_id"].value_counts().head(500).index
ratings = ratings[ratings["product_id"].isin(top_products)]

user_counts = ratings["user_id"].value_counts()
ratings = ratings[
    ratings["user_id"].isin(user_counts[user_counts >= 3].index)
]

product_ids = set(ratings["product_id"])
products = []

for item in read_gzip(
    RAW / "meta_Cell_Phones_and_Accessories.json.gz"
):
    if item.get("asin") not in product_ids:
        continue

    title = item.get("title", "")
    categories = item.get("categories") or []
    category = " | ".join(
        str(name) for group in categories for name in group
    )

    products.append({
        "product_id": item["asin"],
        "title": title,
        "brand": item.get("brand", ""),
        "category": category
    })

products = pd.DataFrame(products)
products = products.drop_duplicates("product_id")

ratings = ratings[
    ratings["product_id"].isin(products["product_id"])
]
products = products[
    products["product_id"].isin(ratings["product_id"])
]

products.to_csv(DATA / "products.csv", index=False)
ratings.to_csv(DATA / "ratings.csv", index=False)

print("Товаров:", len(products))
print("Пользователей:", ratings["user_id"].nunique())
print("Оценок:", len(ratings))
