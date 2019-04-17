# -*- coding: utf-8 -*-
# Uses python 3.7.1

# This code downloads meanings from categories in spreadthesign for RSL


import re
import requests
from bs4 import BeautifulSoup
import math
import csv


def get_page(url_address):
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36' + \
                 ' (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36'

    r = requests.get(url_address + '?q=&p=' + str(1), headers={'User-agent': user_agent})
    soup = BeautifulSoup(r.content, features="lxml", from_encoding=r.encoding)
    paginate = soup.find("div", class_="col-xs-4 search-pager-desc").get_text()
    num_pages = int(re.search('[0-9]/(.+)?', paginate).group(1))
    iter = int(math.ceil(num_pages / 50))

    word_list = []
    for i in range(1, iter + 1):
        r = requests.get(url_address + '?q=&p=' + str(i), headers={'User-agent': user_agent})
        soup = BeautifulSoup(r.content, features="lxml", from_encoding=r.encoding)
        words = soup.find_all("div", class_="search-result-title")  # list
        ws = [el.get_text() for el in words]
        words_50 = []
        for el in ws:
            res = re.search('\n\n        (.+)?\n', el)
            words_50.append(res.group(1))
        word_list.extend(words_50)

    with open('all_meanings.txt', 'a', encoding='utf-8') as file:
        for word in word_list:
            file.write(word + '\n')


if __name__ == "__main__":
    link = 'https://www.spreadthesign.com/ru.ru/search/by-category/'
    upper_links = [link + '398/zhestovyi-iazyk-dlia-nachinaiushchikh/', link + '1/obshchie-polozheniia/',
                   link + '86/religiia/', link + '13/pedagogika/', link + '28/iazyk/',
                   link + '113/iskusstvo-i-razvlecheniia/', link + '46/sotsialnye-issledovaniia/',
                   link + '99/geografiia-i-puteshestviia/', link + '125/eda-i-napitki/', link + '68/stil-zhizni/',
                   link + '147/sport-i-otdykh/', link + '213/tekhnologiia/',
                   link + '227/kompiuter-i-sovremennye-tekhnologii/', link + '168/nauka/',
                   link + '194/zdorove-i-meditsina/', link + '255/zhesty-malyshei/']

    for l in upper_links:
        get_page(l)
