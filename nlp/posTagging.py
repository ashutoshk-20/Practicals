import spacy

nlp = spacy.load("en_core_web_sm")

text = input("Enter a sentence: ")

doc = nlp(text)

for token in doc:
    print(f"Word: {token.text}, POS: {token.pos_}")