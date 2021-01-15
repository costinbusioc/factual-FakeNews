import pandas as pd

df = pd.read_csv("factual.csv")

subjects_dict = {
    "Coronavirus": "Coronavirus",

    "Finante": "Economie",
    "Economie": "Economie",
    "Turism": "Economie",
    "Mediu de afaceri": "Economie",
    "Transport": "Economie",
    "Social": "Economie",
    "Industrie": "Economie",
    "Agricultura": "Economie",
    "Energie": "Economie",

    "Politica": "Politica",
    "Electoral": "Politica",
    "Prezidentiale": "Politica",
    "Razgandiri": "Politica",

    "Europa": "Externe",
    "Externe": "Externe",

    "Justitie": "Justitie",
    "Aparare": "Justitie",

    "Sanatate": "SanatateMediu",
    "Mediu": "SanatateMediu",

    "Sport": "Social",
    "Educatie": "Social",
    "Munca": "Social",
    "Cultura": "Social",
}

subjects = []
for index, row in df.iterrows():
    if len(row["type"].split()) < 2:
        # DECLARATII
        subjects.append("Politica")
        continue

    subjects.append(subjects_dict[" ".join(row["type"].split()[1:])])

df["subject"] = subjects
df.to_csv("factual.csv")
