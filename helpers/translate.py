import pandas as pd
from googletrans import Translator
from google_trans_new import google_translator
import time

df = pd.read_csv("factual.csv")

translator = google_translator(url_suffix="ro")

translations = []
texts = []
for index, row in df.iterrows():
    texts.append(row["text"])
    translations.append(translator.translate(row["text"], lang_tgt="en", lang_src="ro"))
    time.sleep(5)
    print(index)
    print(translations[-1])

#translations = [t for t in translator.translate(texts, lang_tgt="en", lang_src="ro")]


df["English"] = translations
df.to_csv("factual.csv")