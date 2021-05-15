import pandas as pd

from elasticsearch import Elasticsearch

from detect_nouns import get_nouns, get_org_persons

DOMAIN = "localhost"
PORT = 9200
index = "factual-news"

host = str(DOMAIN) + ":" + str(PORT)

client = Elasticsearch(host)

# all_indices = client.indices.get_alias("*")

def read_csv():
    data = []
    df = pd.read_csv("factual_big.csv")

    for index, row in df.iterrows():
        data.append(
            {
                "text": row["text"].strip(),
                "urls": [url for url in row["validation_resources"].split("\n") if url],
                "context": row["validation_content"].strip(),
            }
        )

    return data


def query_by_field(field, text):
    return {
        "query": {
            "match": {
                f"{field}": {
                    "query": text,
                }
            }
        }
    }

def run_query(query):
    resp = client.search(
        index=index,
        body=query,
    )
    return resp["hits"]


def get_all_docs():
    doc_count = 0
    match_all = {"size": 100, "query": {"match_all": {}}}

    # make a search() request to get all docs in the index
    resp = client.search(
        index=index,
        body=match_all,
        scroll="2s",  # length of time to keep search context
    )

    # keep track of pass scroll _id
    old_scroll_id = resp["_scroll_id"]

    es_urls = []
    while len(resp["hits"]["hits"]):
        resp = client.scroll(
            scroll_id=old_scroll_id,
            scroll="2s",  # length of time to keep search context
        )

        # check if there's a new scroll ID
        if old_scroll_id != resp["_scroll_id"]:
            print("NEW SCROLL ID:", resp["_scroll_id"])

        # keep track of pass scroll _id
        old_scroll_id = resp["_scroll_id"]

        # print the response results
        print("\nresponse for index:", index)
        print("_scroll_id:", resp["_scroll_id"])
        print('response["hits"]["total"]["value"]:', resp["hits"]["total"]["value"])

        # iterate over the document hits for each 'scroll'
        for doc in resp["hits"]["hits"]:
            print("\n", doc["_id"], doc["_source"]["url"])
            doc_count += 1
            print("DOC COUNT:", doc_count)
            es_urls.append(doc["_source"]["url"])

    # print the total time and document count at the end
    print("\nTOTAL DOC COUNT:", doc_count)
    es_urls.sort()
    print(len(es_urls))


not_found = 0

"""
for urls_list in get_urls():
    urls = filter_urls(urls_list)
    for url in urls:
        if url in es_urls:
            print(url)
        else:
            not_found += 1

        if not_found % 1000 == 0:
            print(f"Not found: {not_found}")
    for url in urls:
        hits = query_by_url(url)
        print(hits)
        if hits["hits"]:
        break
"""

'''
url = "https://www.libertatea.ro/stiri/teste-coronavirus-2915434"
resp = query_by_field("url", url)
for hit in resp["hits"]:
    print(hit["_score"])
    print(hit["_source"]["url"])
'''


data = read_csv()

for i in range(10):
    entry = data[i]
    print(entry["text"])

    nouns = get_nouns(entry["text"])
    orgs_pers = get_org_persons(entry["text"])

    print(nouns)
    print(orgs_pers)

    query = query_by_field("maintext", entry["text"])
    resp = run_query(query)

    unique_entries = []
    selected_urls = {}
    selected_titles = {}

    for hit in resp["hits"]:
        url = hit["_source"]["url"]
        title = hit["_source"]["title"]

        if len(unique_entries) == 3:
            break

        if selected_urls.get(url) or selected_titles.get(title):
            continue

        unique_entries.append(hit)
        selected_urls[url] = 1
        selected_titles[title] = 1

    for hit in unique_entries:
        print(hit["_score"])
        print(hit["_source"]["url"])
        print(hit["_source"]["title"])
        print(hit["_source"]["maintext"])
        print("\n")

    print("=========")
