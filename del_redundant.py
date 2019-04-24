# Klezovich 10.02.2019
# python 3.7.1

# This code deletes meanings which are longer than three words
# and deletes repeated words

with open('cleaned2328.txt', 'r', encoding='utf-8') as file:
    s = file.readlines()
    words = [el.strip('\n') for el in s]

print(len(words))
set_words = []
for i, w in enumerate(words):
    i += 1
    if w not in set_words:
        set_words.append(w)
print(len(set_words))

setclean = []
for el in set_words:
    if len(el.split(' ')) <= 2:
        setclean.append(el)

print(len(setclean))

with open('ohne_meanings.txt', 'w', encoding='utf-8') as file:
    for word in setclean:
        file.write(word + '\n')
