# Import necessary libraries
import nltk
from nltk import bigrams, trigrams
from nltk.corpus import reuters
from collections import defaultdict

# Download necessary NLTK resources
nltk.download('reuters')
nltk.download('punkt')

# Tokenize the text
words = nltk.word_tokenize(' '.join(reuters.words()))

# Create trigrams
tri_grams = list(trigrams(words))

# Build a trigram model
model = defaultdict(lambda: defaultdict(lambda: 0))

# Count frequency of co-occurrence
for w1, w2, w3 in tri_grams:
    model[(w1, w2)][w3] += 1

# Transform the counts into probabilities
for w1_w2 in model:
    total_count = float(sum(model[w1_w2].values()))
    for w3 in model[w1_w2]:
        model[w1_w2][w3] /= total_count

# Function to predict the next word
def predict_next_word(w1, w2, steps = 5):
    """
    Predicts the next word based on the previous two words using the trained trigram model.
    Args:
    w1 (str): The first word.
    w2 (str): The second word.

    Returns:
    str: The predicted next word.
    """
    words  = []
    for _ in range(steps):
        next_word = model[w1, w2]
        if next_word:
            predicted_word = max(next_word, key=next_word.get)  # Choose the most likely next word
            w1 = w2
            w2 = predicted_word
            words.append(predicted_word)
        else:
            return "No prediction available"
    return words

# Example usage
print("He is ",end="")
for word in predict_next_word("He","is"):
    print(word,end=" ")
