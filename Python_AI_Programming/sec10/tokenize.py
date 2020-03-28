# -*- coding: utf-8 -*-

from nltk.tokenize import sent_tokenize, word_tokenize, WordPunctTokenizer

input_text = "Do you know how tokenization works? \
              It's acutually quite interesting!\
              Let's analyze a couple of sentences and figure it out."
              
print("Sentence tokenizer:")
print(sent_tokenize(input_text))

print("Word tokenizer:")
print(word_tokenize(input_text))

print("Word punct tokenizer:")
print(WordPunctTokenizer().tokenize(input_text))