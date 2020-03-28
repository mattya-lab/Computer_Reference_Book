# -*- coding: utf-8 -*-

from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer

input_words = ['writing', 'calves', 'be', 'branded', 'horse', 'randomize', 
               'possibly', 'provision', 'hospital', 'kept', 'scratchy', 'code']

porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer('english')

stemmer_names = ['INPUT WORD', 'PORTER', 'LANCASTER', 'SNOWBALL']
fmt = '{:>16}' * len(stemmer_names)
print(fmt.format(*stemmer_names))
print('=' * 68)

for word in input_words:
    output = [word, porter.stem(word), lancaster.stem(word), snowball.stem(word)]
    print(fmt.format(*output))