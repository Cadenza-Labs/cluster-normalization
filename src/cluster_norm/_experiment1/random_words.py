import json
import random

import nltk
from nltk.corpus import words

# Downloading the words corpus
nltk.download("words")

# Getting a list of English words
english_words = words.words()

# Ensuring uniqueness and randomness, and selecting 128 words
random_unique_words = random.sample(english_words, 128)

# Converting the list to JSON format
random_words_json = {"words": random_unique_words}

# save to file
with open("random_words_eval.json", "w") as f:
    json.dump(random_words_json, f)
