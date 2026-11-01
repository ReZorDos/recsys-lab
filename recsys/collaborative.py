from html import unescape
from pathlib import Path

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


DATA_DIR = Path(__file__).resolve().parent.parent / "data"

ratings = pd.read_csv(DATA_DIR / "train.csv")
products = pd.read_csv(DATA_DIR / "products.csv")

matrix = ratings.pivot_table(
    index="user_id",
    columns="product_id",
    values="rating"
)

filled_matrix = matrix.fillna(0)

similarity = cosine_similarity(filled_matrix)

user_similarity = pd.DataFrame(
    similarity,
    index=matrix.index,
    columns=matrix.index
)

def recommend_for_user(user_id, top_n=5):
    if user_id not in matrix.index:
        return pd.DataFrame()

    similarities = cosine_similarity(
        filled_matrix.loc[[user_id]],
        filled_matrix
    )[0]

    similar_users = (
        pd.Series(similarities, index=matrix.index)
        .drop(user_id)
        .sort_values(ascending=False)
        .head(20)
    )

    similar_users = similar_users[similar_users > 0]

    if similar_users.empty:
        return pd.DataFrame()

    watched = set(
        ratings.loc[
            ratings["user_id"] == user_id,
            "product_id"
        ]
    )

    candidates = ratings[
        ratings["user_id"].isin(similar_users.index)
        & ~ratings["product_id"].isin(watched)
    ].copy()

    if candidates.empty:
        return pd.DataFrame()

    candidates["weight"] = candidates["user_id"].map(
        similar_users
    )
    candidates["weighted_rating"] = (
        candidates["rating"] * candidates["weight"]
    )

    scores = candidates.groupby("product_id").agg(
        weighted_sum=("weighted_rating", "sum"),
        weight_sum=("weight", "sum")
    )

    scores["score"] = (
            scores["weighted_sum"] / scores["weight_sum"]
    )

    scores = scores[scores["score"] >= 3.5]

    scores = scores.sort_values(
        "score", ascending=False
    ).head(top_n)

    result = scores[["score"]].reset_index()

    return result.merge(
        products[["product_id", "title"]],
        on="product_id",
        how="left"
    )


if __name__ == "__main__":
    user_id = ratings["user_id"].value_counts().index[0]

    print("Пользователь:", user_id)
    print("\nРекомендации:")

    recommendations = recommend_for_user(user_id)

    if recommendations.empty:
        print("Не удалось найти рекомендации")
    else:
        for _, item in recommendations.iterrows():
            print(
                f"{unescape(str(item['title']))} "
                f"(оценка: {item['score']:.2f})"
            )
