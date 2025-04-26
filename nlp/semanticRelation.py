import nltk
nltk.download('wordnet')
nltk.download('omv-1.4')
from nltk.corpus import wordnet as wn

def get_semantics_relations(word1,word2):
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)

    if not synsets1 or not synsets2:
        return f"Seamtics Relationships could not be determined for {word1} and {word2}"
    
    synset1 = synsets1[0]
    synset2 = synsets2[0]

    relationships = {
        'Word 1':word1,
        'Word 2':word2,
        'Synset 1 Definition': synset1.definition(),
        'Hypernyms of Word 1': [lemma.name() for hypernym in synset1.hypernyms() for lemma in hypernym.lemmas()],
        'Hyponyms of Word 1': [lemma.name() for hyponym in synset1.hyponyms() for lemma in hyponym.lemmas()],
        'Synset 2 Definition': synset2.definition(),
        'Hypernyms of Word 2': [lemma.name() for hypernym in synset2.hypernyms() for lemma in hypernym.lemmas()],
        'Hyponyms of Word 2': [lemma.name() for hyponym in synset2.hyponyms() for lemma in hyponym.lemmas()], 
        'Similarity Score': synset1.wup_similarity(synset2)      
    }

    return relationships

word1 = input("Enter the first word: ")
word2 = input("Enter the second word: ")

print()
semantic_relation = get_semantics_relations(word1, word2)

for key,value in semantic_relation.items():
    print(f"{key}: {value}\n")