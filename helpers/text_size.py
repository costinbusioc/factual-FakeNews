import re
import pandas as pd

df = pd.read_csv("factual.csv")

SMALL_BIG_TEXT_LIMIT = 35

sizes = []
for index, row in df.iterrows():
    size = len(re.findall(r'\w+', row["text"]))
    if size < SMALL_BIG_TEXT_LIMIT:
        sizes.append("small")
    else:
        sizes.append("big")

df["text_size"] = sizes
df.to_csv("factual.csv")