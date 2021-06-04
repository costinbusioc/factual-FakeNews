from elasticsearch import Elasticsearch

from helpers import print_hit, get_unique_entries

DOMAIN = "localhost"
PORT = 9200
index = "factual-news"

host = str(DOMAIN) + ":" + str(PORT)

client = Elasticsearch(host)

def run_query(query):
    resp = client.search(
        index=index,
        body=query,
    )
    return resp["hits"]

query = {
    "query": {
        "bool": {
          "should": [
            {
                "match": {
                    f"maintext": {
                        "query": "PNL",
                        "boost": 2,
                    }
                }
            },
            {
                "match": {
                  f"maintext": {
                      "query": "parlamentari pensiilor",
                      "operator": "and",
                      "boost": 2,
                  }
                },
            },
            {
                "match": {
                  f"maintext": {
                      "query": "PNL a fost singurul partid care a votat împotriva introducerii pensiilor speciale pentru parlamentari și am avut această poziție de-a lungul ultimilor ani.",
                  }
                },
            },
          ]
        }
      }
    }

resp = get_unique_entries(run_query(query))

for hit in resp:
    print_hit(hit)

