# Klezovich 10.02.2019
# python 3.7.1

# This program downloads sign videos by their translations from spreadthesign
# The list of translations is read from txt file

import sys, os
sys.path.insert(0, os.path.abspath('..'))
from lingcorpora.corpus import Corpus


def download(words):
    corp = Corpus('sls')
    results = corp.search(words, query_language='ru.ru', subcorpus='ru.ru', only_link=False)
    for result in results:
        for i, target in enumerate(result):
            return target.transl

if __name__ == "__main__":
    with open('final.txt', 'r', encoding='utf-8') as file:
        s = file.readlines()
        words = [el.strip('\n') for el in s]
    print(len(words))
    download(words)
