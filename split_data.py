from pathlib import Path
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"

ratings = pd.read_csv(DATA_DIR / "ratings.csv")

ratings = ratings.sort_values(["user_id", "timestamp"])

train_parts = []
test_parts = []

for user_id, group in ratings.groupby("user_id"):
    n = len(group)

    if n < 2:
        continue

    train_size = int(n * 0.8)
    train_size = max(1, min(train_size, n - 1))

    train_parts.append(group.iloc[:train_size])
    test_parts.append(group.iloc[train_size:])

train = pd.concat(train_parts, ignore_index=True)
test = pd.concat(test_parts, ignore_index=True)

known_products = set(train["product_id"])
test = test[test["product_id"].isin(known_products)].copy()

train.to_csv(DATA_DIR / "train.csv", index=False)
test.to_csv(DATA_DIR / "test.csv", index=False)

print("\nПользователей в Train:", train["user_id"].nunique())
print("Пользователей в Test:", test["user_id"].nunique())
print("Товаров в Train:", train["product_id"].nunique())
print("Товаров в Test:", test["product_id"].nunique())
