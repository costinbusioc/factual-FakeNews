import requests
from bs4 import BeautifulSoup

import pandas as pd

base_url = "https://www.factual.ro/toate-declaratiile/"

authors = []
categories = []
labels = []
texts = []
verificare = []
verificare_1 = []
verificate_last = []
resources = []

page = requests.get(base_url)
soup = BeautifulSoup(page.content, "html.parser")

statements = soup.find_all("div", class_="col-md-4")

for i, statement in enumerate(statements):
    print(f"Processing statement {i}/{len(statements)}")

    category = statement.find("div", class_="titludeclaratie2 upper").text

    author = statement.find("div", class_="candidat").text
    label = statement.find("p", class_="text_btn").text

    url = statement.find("a")["href"]

    analysis_page = requests.get(url)
    analysis = BeautifulSoup(analysis_page.content, "html.parser")

    text = analysis.find("h1", class_="titludeclaratie")
    if not text:
        continue
    text = text.text

    verificare_section = analysis.find_all("div", class_="col-md-12 corp")[1]
    paragraphs = verificare_section.find_all("p")

    current_verificare = []
    current_resources = []

    if not paragraphs:
        verificare_1.append("")
        verificate_last.append("")

    for j, p in enumerate(paragraphs):
        if j == 0:
            # Skip the first redundant paragraph
            continue

        current_verificare.append(p.text)

        if j == 1:
            verificare_1.append(p.text)

        if j == len(paragraphs) - 1:
            verificate_last.append(p.text)

    for resource in verificare_section.find_all("a"):
        current_resources.append(resource["href"])

    categories.append(category)
    authors.append(author)
    texts.append(text)
    labels.append(label)

    if current_verificare:
        verificare.append("\n\n".join(current_verificare))
    else:
        verificare.append("")

    if current_resources:
        resources.append("\n\n".join(current_resources))
    else:
        resources.append("")

    print(verificare_1)
    print(verificate_last)
    print(resources)
    print("\n\n\n")


cols = [authors, texts, categories, labels, verificare_1, verificate_last, resources, verificare]
col_names = ["author", "text", "category", "label", "first_validation_par", "last_validation_par", "validation_resources", "validation_content"]

df = pd.DataFrame(cols)
df = df.transpose()

with open("factual.csv", "w", encoding="utf-8") as f:
    df.to_csv(f, header=col_names)
