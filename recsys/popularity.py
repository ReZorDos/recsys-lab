
from html import unescape
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "data"

ratings = pd.read_csv(DATA_DIR / "train.csv")
products = pd.read_csv(DATA_DIR / "products.csv")


def recommend_popular(top_n=5):
    stats = ratings.groupby("product_id").agg(
        avg_rating=("rating", "mean"),
        rating_count=("rating", "count")
    ).reset_index()

    global_mean = ratings["rating"].mean()

    min_reviews = 10

    stats["score"] = (
        (stats["rating_count"] * stats["avg_rating"]
         + min_reviews * global_mean)
        / (stats["rating_count"] + min_reviews)
    )

    stats = stats.sort_values(
        "score", ascending=False
    ).head(top_n)

    return stats.merge(
        products[["product_id", "title"]],
        on="product_id",
        how="left"
    )


if __name__ == "__main__":
    print("Самые популярные гаджеты:\n")
    recommendations = recommend_popular()
    for _, item in recommendations.iterrows():
        print(
            f"{unescape(str(item['title']))} "
            f"(рейтинг: {item['score']:.2f}, "
            f"отзывов: {item['rating_count']})"
        )
