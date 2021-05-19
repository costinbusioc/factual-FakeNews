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
                          "query": "cadre medicale vaccinat",
                          "operator": "and",
                          "boost": 2,
                      }
                  },
              },
            {
                "match": {
                  f"maintext": {
                    "query": "Majoritatea cadrelor medicale s-au vaccinat."
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

