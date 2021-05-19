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
                          "query": "Uniunea Europeană",
                          "boost": 2,
                      }
                  },
              },
              {
                  "match": {
                      f"maintext": {
                          "query": "economie salariu crestere",
                          "operator": "and",
                          "boost": 2,
                      }
                  },
              },
              {
                "match": {
                  f"maintext": {
                    "query": "Avem în momentul de față cea mai mică creștere a salariului minim pe economie din Uniunea Europeană."
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

