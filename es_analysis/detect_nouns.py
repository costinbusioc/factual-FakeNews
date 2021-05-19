import spacy

nlp = spacy.load("ro_core_news_lg")

def get_nouns(sentence):
    nouns = []
    doc = nlp(sentence)
    for token in doc:
        if token.pos_ == "NOUN":
            nouns.append(token.text)

    return nouns

def get_org_persons(sentence):
    orgs_pers = []
    doc = nlp(sentence)
    for ent in doc.ents:
        print(ent)
        print(ent.label_)
        if ent.label_ in ["ORGANIZATION", "PERSON", "GPE"]:
            orgs_pers.append(ent.text)

    return orgs_pers