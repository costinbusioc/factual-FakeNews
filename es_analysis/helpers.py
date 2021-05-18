def print_hit(hit):
    print(hit["_score"])
    print(hit["_source"]["url"])
    print(hit["_source"]["title"])
    print(hit["_source"]["maintext"])
    print("\n")