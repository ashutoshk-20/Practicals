import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
import spacy

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
# spacy.cli.download("en_core_web_sm")

prefixes = ['un', 're', 'dis', 'pre', 'mis', 'in', 'im', 'non', 'over', 'under', 'inter', 'sub', 'trans', 'super', 'semi', 'anti', 'co', 'de']
suffixes = ['ing', 'ed', 'ness', 'ly', 'able', 'ible', 'ment', 'tion', 'sion', 'er', 'or', 'ist', 'al', 'ous', 'ive', 'ity', 'y', 'en', 'ize', 'ise']

lemmatizer = WordNetLemmatizer()

nlp = spacy.load("en_core_web_sm")

# Function to determine POS from spaCy
def get_wordnet_pos_spacy(text):
    doc = nlp(text)
    for token in doc:
        return token.pos_

# Function to convert POS tag to WordNet POS format (for NLTK)
def get_wordnet_pos(tag):
    if tag.startswith('J'): return wordnet.ADJ
    elif tag.startswith('V'): return wordnet.VERB
    elif tag.startswith('N'): return wordnet.NOUN
    elif tag.startswith('R'): return wordnet.ADV
    return wordnet.NOUN

# Fix spelling for "happi" → "happy" after suffix removal (e.g., 'ness')
def fix_root_spelling_on_delete(root, suffix):
    if suffix == 'ness' and root.endswith('i'):
        return root[:-1] + 'y'  # Convert 'happi' to 'happy'
    return root

# Fix spelling for "happy" → "happi" before suffix add (e.g., 'ness')
def fix_root_spelling_on_add(root, suffix):
    if suffix == 'ness' and root.endswith('y'):
        return root[:-1] + 'i'  # Convert 'happy' to 'happi'
    return root

# Adjust handling of adverbs like "happily" → "happi"
def adjust_for_adverbs(root, suffix):
    if suffix == 'ly' and root.endswith('i'):
        return root[:-1] + 'y' # Convert 'happily' to 'happi'
    return root

def adjusst_for_adverbs_add(root, suffix):
    if suffix == 'ly' and root.endswith('y'):
        return root[:-1] + 'i'  # Convert 'happi' to 'happy'
    return root

# Morphology table function
def morphology_table(word):
    original = word
    prefix = ''
    suffix = ''
    core = word

    # Detect prefix
    for p in prefixes:
        if word.startswith(p):
            prefix = p
            core = word[len(p):]
            break

    # Detect suffix
    for s in suffixes:
        if core.endswith(s):
            suffix = s
            core = core[:-len(s)]
            break

    # Fix spelling of root
    root = fix_root_spelling_on_delete(core, suffix)
    root = adjust_for_adverbs(root, suffix)  # Special case for "ly"

    # Display Table
    print(f"\n[NLTK+Smart] Add/Delete Table for: {original}")
    print("-" * 42)
    print(f"{'Action':<8} | {'Morpheme':<10} | {'Result'}")
    print("-" * 42)
    print(f"{'Start':<8} | {'-':<10} | {original}")

    # Delete suffix
    if suffix:
        step1 = original[:-len(suffix)]
        print(f"{'Delete':<8} | {suffix:<10} | {prefix + core}")

    # Delete prefix
    if prefix:
        step2 = core
        print(f"{'Delete':<8} | {prefix:<10} | {core}")

    # Show root
    print(f"{'Root':<8} | {'-':<10} | {root}")

    # Add prefix
    if prefix:
        with_prefix = prefix + root
        print(f"{'Add':<8} | {prefix:<10} | {with_prefix}")
    else:
        with_prefix = root

    # Add suffix
    if suffix:
        fixed_root = fix_root_spelling_on_add(root, suffix)
        fixed_root = adjusst_for_adverbs_add(fixed_root, suffix)
        final_word = prefix + fixed_root + suffix
        print(f"{'Add':<8} | {suffix:<10} | {final_word}")

# Test with words
morphology_table("unhappiness")
morphology_table("happily")
morphology_table("disagreeable")