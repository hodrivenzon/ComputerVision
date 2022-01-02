import pandas as pd
import requests
from bs4 import BeautifulSoup
import csv

def main():
    # URL = "https://www.yad2.co.il/realestate/forsale?topArea=101&area=87&city=0240"
    # URL = "https://www.yad2.co.il/"
    URL = "https://www.walla.co.il/"
    r = requests.get(URL)
    soup = BeautifulSoup(r.text, 'html.parser')
    print(soup.prettify())
    quotes = []
    results = []
    for num in range(10):
        address = soup.find('span', id=f'feed_item_{num}_title').text
        price = soup.find('div', id=f'feed_item_{num}_price').text
        rooms = soup.find('span', attrs ={"id": f'data_rooms_{num}', "class":'val'}).text
        floor = soup.find('span', attrs={"id": f'data_floor_{num}', "class": 'val'}).text
        size = soup.find('span', attrs={"id": f'data_SquareMeter_{num}', "class": 'val'}).text
        results.append({"address": address,
                        "price": price,
                        "rooms": rooms,
                        "floor": floor,
                        "size": size})
    print()
    # table = soup.find('div', attrs={'id': 'all_quotes'})
    #
    # for row in table.findAll('div',
    #                          attrs={'class': 'col-6 col-lg-3 text-center margin-30px-bottom sm-margin-30px-top'}):
    #     quote = {}
    #     quote['theme'] = row.h5.text
    #     quote['url'] = row.a['href']
    #     quote['img'] = row.img['src']
    #     quote['lines'] = row.img['alt'].split(" #")[0]
    #     quote['author'] = row.img['alt'].split(" #")[1]
    #     quotes.append(quote)
    #
    # filename = 'inspirational_quotes.csv'
    # with open(filename, 'w', newline='') as f:
    #     w = csv.DictWriter(f, ['theme', 'url', 'img', 'lines', 'author'])
    #     w.writeheader()
    #     for quote in quotes:
    #         w.writerow(quote)

if __name__ == '__main__':
    main()