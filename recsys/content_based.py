from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_DIR = Path(__file__).resolve().parent.parent / "data"

products = pd.read_csv(DATA_DIR / "products.csv")
products = products.fillna("")

products["text"] = (
    products["title"] + " " +
    products["brand"] + " " +
    products["category"]
)

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(products["text"])


def recommend_similar(product_id, top_n=5):
    matches = products.index[
        products["product_id"] == product_id
    ].tolist()

    if not matches:
        return pd.DataFrame()

    index = matches[0]

    similarities = cosine_similarity(
        tfidf_matrix[index],
        tfidf_matrix
    ).flatten()

    similar_indices = similarities.argsort()[::-1]
    similar_indices = [
        i for i in similar_indices if i != index
    ][:top_n]

    result = products.iloc[similar_indices][
        ["product_id", "title", "brand"]
    ].copy()

    result["similarity"] = similarities[similar_indices]

    return result


if __name__ == "__main__":
    product = products.iloc[0]
    print("Выбранный товар:", product["title"])
    recommendations = recommend_similar(product["product_id"])

    for _, item in recommendations.iterrows():
        print(
            f"{item['title']} "
            f"(сходство: {item['similarity']:.2f})"
        )
