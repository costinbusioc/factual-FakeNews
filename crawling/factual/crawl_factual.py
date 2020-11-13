import requests
from bs4 import BeautifulSoup

import pandas as pd

#base_url = "https://www.factual.ro/toate-declaratiile/"
base_url = "https://www.factual.ro/coronavirus/"

authors = []
categories = []
labels = []
texts = []

page = requests.get(base_url)
soup = BeautifulSoup(page.content, "html.parser")

#statements = soup.find_all("div", class_="col-md-4")
statements = soup.find_all("div", class_="col-md-4 paddingleft0")

for i, statement in enumerate(statements):
    print(f"Processing statement {i}/{len(statements)}")

    #category = statement.find("div", class_="titludeclaratie2 upper").text
    category = statement.find("div", class_="titludeclaratie2").text

    author = statement.find("div", class_="candidat").text
    label = statement.find("p", class_="text_btn").text

    url = statement.find("a")["href"]

    analysis_page = requests.get(url)
    analysis = BeautifulSoup(analysis_page.content, "html.parser")

    text = analysis.find("h1", class_="titludeclaratie").text

    categories.append(category)
    authors.append(author)
    texts.append(text)
    labels.append(label)


cols = [authors, texts, categories, labels]
col_names = ["author", "text", "category", "label"]

df = pd.DataFrame(cols)
df = df.transpose()

#with open("factual.csv", "w", encoding="utf-8") as f:
with open("factual_coronavirus.csv", "w", encoding="utf-8") as f:
    df.to_csv(f, header=col_names)
