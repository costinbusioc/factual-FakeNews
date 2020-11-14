from datetime import date
import requests
from bs4 import BeautifulSoup

import re
import csv
import sys
import json
import os
import glob
import pandas as pd

MAX_PAGE_NO = 250

headers = {'User-Agent': 
           'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'}

def write_csv(filename, col_names, cols):
    df = pd.DataFrame(cols)
    df = df.transpose()

    with open(filename, 'w', encoding='utf-8', newline = '\n') as f:
            df.to_csv(f, header=col_names)

def get_list_(input_table, column_name):
	df = pd.read_csv(input_table)
	return df[column_name].tolist()

def convert_category(category):
	result = 0
	if category == "mostly-true" or category == "half-true":
		result = 1
	if category == "barely-true":
		result = 2
	if category == "false" or category == "pants-fire":
		result = 3
	return result


root_link = "https://www.politifact.com/factchecks/list/?"
categories = ["true", "mostly-true", "half-true", "barely-true", "false", "pants-fire"]

cols_name = ['source', 'location', 'text', 'detective', 'category', 'label']
sources = []
locations = []
texts = []
detectives = []
all_categories = []
all_labels = []
statistics = ''

for category in categories:
	category_no = 0
	for link_no in range(1, MAX_PAGE_NO):
		link = root_link + "page=" + str(link_no) + "&ruling=" + category
		
		print ("Link: " + link)

		page = requests.get(link, headers = headers)
		soup = BeautifulSoup(page.content, "html.parser")
		tag = soup.find('section', class_="o-listicle")
		if tag == None:
			if soup.find(text='Next') == None:
				break
			else:
				continue
		tag = tag.findNext('article')
		rows = tag.find_all('li')

		for row in rows:
			meta = row.findNext('div', class_="m-statement__meta")
			source = meta.find('a').text.strip()
			location = meta.find('div').text.strip()
			content = meta.findNext('div', class_="m-statement__content")
			text = content.find('a').text.strip()
			detective = content.findNext('footer').text.strip()

			category_no += 1
			sources.append(source)
			locations.append(location)
			texts.append(text)
			detectives.append(detective)
			all_categories.append(category)
			all_labels.append(convert_category(category))

			# print (source)
			# print (location)
			# print (text)
			# print (detective)
			# print ("==================")
	statistics += category + ": " + str(category_no) + "\n"

cols = [sources, locations, texts, detectives, all_categories, all_labels]
write_csv('datasetPolitical.csv', cols_name, cols)

print (statistics)


		
