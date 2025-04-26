import nltk
from nltk import ngrams
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

nltk.download('punkt_tab')

text = "Betty bought butter but the butter was bitter. So Betty bought better butter to make the bitter butter better."

tokens = word_tokenize(text.lower())

unigram_list = list(ngrams(tokens,1))
bigram_list = list(ngrams(tokens,2))
trigram_list = list(ngrams(tokens,3))

unigram_freq = FreqDist(unigram_list)
bigram_freq = FreqDist(bigram_list)
trigram_freq = FreqDist(trigram_list)

print("\nUnigrams : ")
for unigram,freq in unigram_freq.items():
    print(f"{unigram}: {freq}")

print("\nBigrams : ")
for bigram,freq in bigram_freq.items():
    print(f"{bigram}: {freq}")

print("\ntrigrams : ")
for trigram,freq in trigram_freq.items():
    print(f"{trigram}: {freq}")
