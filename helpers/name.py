import pandas as pd
import spacy

nlp = spacy.load("ro_core_news_sm") # or _md or _lg models

df = pd.read_csv("factual.csv")


def text_contains_name(spacy_doc):
    for ent in spacy_doc.ents:
        if ent.label_ in ["ORGANIZATION", "PERSON"]:
            return True

    return False

contains_name = []
for index, row in df.iterrows():
    text = row["text"]
    doc = nlp(text)

    if text_contains_name(doc):
        contains_name.append("yes")
    else:
        contains_name.append("no")

df["contains_name"] = contains_name
df.to_csv("factual.csv")