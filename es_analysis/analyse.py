import pandas as pd

from elasticsearch import Elasticsearch

DOMAIN = "localhost"
PORT = 9200
index = "factual-news"

host = str(DOMAIN) + ":" + str(PORT)

client = Elasticsearch(host)

# all_indices = client.indices.get_alias("*")

# keep track of the number of the documents returned
doc_count = 0

match_all = {"size": 100, "query": {"match_all": {}}}


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


def query_by_url(url):
    match_url = {
        "query": {
            "match": {
                "url": url,
            }
        }
    }

    resp = client.search(
        index=index,
        body=match_url,
    )
    return resp["hits"]


def get_all_docs():
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

url = "https://www.libertatea.ro/stiri/teste-coronavirus-2915434"
resp = query_by_url(url)
for hit in resp["hits"]:
    print(hit["_score"])
    print(hit["_source"]["url"])
