import pandas as pd
from googletrans import Translator

df = pd.read_csv("factual.csv")

translator = Translator()

translations = []
for index, row in df.iterrows():
    translation = translator.translate(row["text"], dest="en", src="ro").text
    translations.append(translation)

df["English"] = translations
df.to_csv("factual.csv")