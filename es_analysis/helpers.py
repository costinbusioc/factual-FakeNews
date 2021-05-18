def print_hit(hit):
    print(hit["_score"])
    print(hit["_source"]["url"])
    print(hit["_source"]["title"])
    print(hit["_source"]["maintext"])
    print("\n")

def get_unique_entries(resp):
    unique_entries = []
    selected_urls = {}
    selected_titles = {}

    for hit in resp["hits"]:
        url = hit["_source"]["url"]
        title = hit["_source"]["title"]

        if selected_urls.get(url) or selected_titles.get(title):
            continue

        unique_entries.append(hit)
        selected_urls[url] = 1
        selected_titles[title] = 1

    return unique_entries