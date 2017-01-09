#!/usr/bin/python
#coding: utf-8

"""
Token based on char or word.
"""

from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import sent_tokenize
import re
stemmer = SnowballStemmer("english")

def tokenize(text, analyzer="char", stopwords = None):
	text = re.sub(r'(\d+)[,\'](\d+)', r'\1\2', text)
	text = re.sub(r'\W+', ' ', text)
	return [stemmer.stem(x) for x in word_tokenize(text) if len(x) > 1 or x.isnumeric()]

def fetch_sents(text):
    return sent_tokenize(text)


