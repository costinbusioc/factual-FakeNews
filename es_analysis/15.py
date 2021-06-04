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
                "match_phrase": {
                    f"maintext": {
                        "query": "Valeriu Stoica",
                        "boost": 2,
                    }
                }
            },
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
                      "query": "ministru pensiile",
                      "operator": "and",
                      "boost": 2,
                  }
                },
            },
            {
                "match": {
                  f"maintext": {
                      "query": "Nu în 2010 s-au inventat pensiile speciale. Pensiile speciale s-au inventat în 2005 de către Valeriu Stoica, membru PNL, ministru al Justiției din partea PNL. Atunci s-au inventat pensiile speciale.",
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

