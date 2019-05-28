import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
#from IPython.display import display

titles = []
written_times = []
first_lines = []

for i in range(31,1014):
    url = "https://uk.reuters.com/news/archive/GCA-ForeignExchange?view=page&page={}&pageSize=10".format(i)
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')

    for title in soup.find_all('h3', {'class':'story-title'}):
        titles.append(title.string.strip())

    for written_time in soup.find_all('span', {'class':'timestamp'}):
        written_times.append(written_time.string.strip())


    try:
        for first_line in soup.find_all('p'):
            first_lines.append(first_line.string.strip())
    except:
        pass

    titles = titles[:len(titles)-3]
    first_lines = first_lines[:len(first_lines)-2]
    if(i%50==0):
        print(i)

reuters_word_headlines = pd.DataFrame({'title':titles, 'time':written_times, 'headline content':first_lines})
reuters_word_headlines = reuters_word_headlines.loc[:,('time', 'title','headline content')]

reuters_word_headlines.to_csv(path_or_buf='/home/tushar/Desktop/Python/Python3/tradegame-master/output.csv',index=False)
