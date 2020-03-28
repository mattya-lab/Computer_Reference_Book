# -*- coding: utf-8 -*-

from nltk.stem import WordNetLemmatizer

input_words = ['writing', 'calves', 'be', 'branded', 'horse', 'randomize', 
               'possibly', 'provision', 'hospital', 'kept', 'scratchy', 'code']

lemmatizer = WordNetLemmatizer()

lemmatizer_names = ['INPUT WORD', 'NOUN LEMMATIZER', 'VERB LEMMATIZER']
fmt = '{:>24}' * len(lemmatizer_names)
print(fmt.format(*lemmatizer_names))
print('=' * 75)

for word in input_words:
    output = [word, lemmatizer.lemmatize(word, pos='n'),
                    lemmatizer.lemmatize(word, pos='v')]
    print(fmt.format(*output))