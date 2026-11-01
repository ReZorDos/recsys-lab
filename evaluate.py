
import pandas as pd

from recsys.content_based import recommend_similar
from recsys.collaborative import recommend_for_user
from recsys.popularity import recommend_popular


train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
products = pd.read_csv("data/products.csv")

K = 5

users = test[test["rating"] >= 4]["user_id"].unique()[:50]

popular = recommend_popular(top_n=len(products))


def get_recommendations(user_id, method):
    history = train[train["user_id"] == user_id]
    seen = set(history["product_id"])

    if method == "content":
        liked = history[history["rating"] >= 4]

        if liked.empty:
            return []

        product_id = liked.iloc[-1]["product_id"]
        result = recommend_similar(product_id, top_n=50)

    elif method == "collaborative":
        result = recommend_for_user(user_id, top_n=K)

    else:
        result = popular

    return [
        pid for pid in result["product_id"]
        if pid not in seen
    ][:K]


def evaluate(method):
    precisions = []
    recalls = []
    recommended_items = set()

    for user_id in users:
        relevant = set(
            test[
                (test["user_id"] == user_id)
                & (test["rating"] >= 4)
            ]["product_id"]
        )

        if not relevant:
            continue

        recommendations = get_recommendations(user_id, method)
        recommended_items.update(recommendations)

        hits = len(set(recommendations) & relevant)

        precisions.append(hits / K)
        recalls.append(hits / len(relevant))

    if not precisions:
        return 0, 0, 0

    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    coverage = len(recommended_items) / len(products)

    return precision, recall, coverage


methods = {
    "Content-based": "content",
    "Collaborative": "collaborative",
    "Popularity-based": "popularity"
}

print(f"Оценка рекомендаций для {len(users)} пользователей\n")

for name, method in methods.items():
    precision, recall, coverage = evaluate(method)

    print(name)
    print(f"Precision@5: {precision:.4f}")
    print(f"Recall@5:    {recall:.4f}")
    print(f"Coverage:    {coverage:.4f}")
    print()
