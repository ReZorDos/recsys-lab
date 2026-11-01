
from html import unescape

from recsys.content_based import products, recommend_similar
from recsys.collaborative import ratings, recommend_for_user
from recsys.popularity import recommend_popular


def show_recommendations(recommendations):
    if recommendations.empty:
        print("Рекомендации не найдены")
        return

    for _, item in recommendations.iterrows():
        print("-", unescape(str(item["title"])))


def main():
    print("Рекомендательная система гаджетов")
    print("1. Похожие товары (Content-based)")
    print("2. Персональные рекомендации (Collaborative)")
    print("3. Популярные товары (Popularity-based)")

    choice = input("\nВыберите алгоритм (1-3): ")

    if choice == "1":
        print("\nПримеры товаров:")
        for _, item in products.head(5).iterrows():
            print(item["product_id"], "-", unescape(str(item["title"])))

        product_id = input("\nВведите ID товара: ").strip()

        recommendations = recommend_similar(product_id)

    elif choice == "2":
        user_id = ratings["user_id"].value_counts().index[0]

        print("\nПользователь:", user_id)
        recommendations = recommend_for_user(user_id)

    elif choice == "3":
        recommendations = recommend_popular()

    else:
        print("Неверный выбор")
        return

    print("\nРекомендации:")
    show_recommendations(recommendations)


if __name__ == "__main__":
    main()
