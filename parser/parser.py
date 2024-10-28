import json
import os
import re

from bs4 import BeautifulSoup
import requests


def parse_page(link, author):
    page = requests.get(link)

    data = BeautifulSoup(page.text, "html.parser")

    author_header = data.find('h2').next.next
    if author != author_header:
        return

    title_header = data.find('h2').text.replace(author_header, '').strip()
    year = data.find('table').find_all('li')[2].text[5:9]

    genres = None
    genre_tag = data.find('table').find_all('li')[4]
    if genre_tag:
        genres = [a.text for a in genre_tag.find_all('a')]

    annotation = None
    annotation_tag = data.find('table').find_all('li')[-1].find('i')
    if annotation_tag:
        annotation = annotation_tag.text

    text = data.find('noindex').text
    if text is not None:
        result_dir = '../data'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        with open(f"../data/{author_header}.{title_header}.{year}.txt", 'w', encoding='utf-8') as file:
            text = text.replace(" ", " ")
            cleaned_text = re.sub(r'\n+', '\n', text)
            file.write(cleaned_text)

        with open(f"../data/{author_header}.{title_header}.{year}.json", 'w', encoding='utf-8') as json_file:
            json.dump({
                'link': link,
                'author': author_header,
                'title': title_header,
                'year': year,
                'annotation': annotation,
                'genres': genres
            }, json_file, ensure_ascii=False, indent=2)
        print(f"Сохранен '{author_header} - {title_header}'")


def parse_authors():
    urls = []

    with open('urls.txt', 'r', encoding='UTF-8') as file:
        while line := file.readline():
            urls.append(line.rstrip())

    for url in urls:
        author_page = requests.get(url)
        author_data = BeautifulSoup(author_page.text, "html.parser")

        author_name = author_data.find('title').text[17:].split('.')[0]

        a_tags = author_data.find('dl').find_all('a')
        for a in a_tags:
            href = a.get('href')
            if href and href.startswith('text'):
                try:
                    parse_page(url + href, author_name)
                except Exception as e:
                    print(f"Ошибка в {url + href}", e)


if __name__ == '__main__':
    parse_authors()
