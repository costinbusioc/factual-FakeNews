from detect_nouns import get_nouns, get_org_persons

def print_hit(hit):
    print(hit["_score"])
    print("\n")
    # print(hit["_source"]["url"])
    # print(hit["_source"]["title"])
    print(hit["_source"]["maintext"])
    print("\n")

def get_unique_entries(resp):
    unique_entries = []
    selected_urls = {}
    selected_titles = {}
    selected_texts = {}

    for hit in resp["hits"]:
        url = hit["_source"]["url"]
        title = hit["_source"]["title"]
        text = hit["_source"]["maintext"]

        if selected_urls.get(url) or selected_titles.get(title) or selected_texts.get(text):
            continue

        unique_entries.append(hit)
        selected_urls[url] = 1
        selected_titles[title] = 1
        selected_texts[text] = 1

    return unique_entries

def compose_match_phrase_from_word(word, boost=2):
    return {
        "match_phrase": {
            "maintext": {
                "query": word,
                "boost": boost,
            }
        }
    }

def compose_match_from_words_list(words, boost=2):
    return {
        "match": {
            "maintext": {
                "query": " ".join(words),
                "operator": "and",
                "boost": boost,
            }
        }
    }

def compose_query_by_field(text):
    return {
        "match": {
            "maintext": {
                "query": text,
            }
        }
    }

def compose_should_query(queries):
    return {
      "query": {
        "bool": {
          "should": queries
        }
      }
    }

def compute_query_1(entry_text):
    nouns = get_nouns(entry_text)
    orgs_pers = get_org_persons(entry_text)

    print(f"Nouns: {nouns}")
    print(f"Orgs pers: {orgs_pers}")

    should_queries = []
    for org_pers in orgs_pers:
        should_queries.append(compose_match_phrase_from_word(org_pers))

    text_query = compose_query_by_field(entry_text)
    print(text_query)
    nouns_query = compose_match_from_words_list(nouns)

    should_queries.append(nouns_query)
    should_queries.append(text_query)

    return compose_should_query(should_queries)
