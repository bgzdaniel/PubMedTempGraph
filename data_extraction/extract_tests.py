import pandas as pd
df = pd.read_csv("../data/studies.csv")
print(df["Abstract"].isna().sum() / len(df))