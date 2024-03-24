import requests
from urllib.request import urlopen
from bs4 import BeautifulSoup
from utils import *
from scrapper import *
import pandas as pd
import re
import os
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from tqdm import tqdm
import warnings


def main():
    base_url = 'https://primeminister.gr/category/activity/statements'
    csv_path = 'dataset_statements.csv'
    speeches_links_file = 'statements_url.log'

    links = scrapping(base_url, speeches_links_file)

    # Retrieve content per url
    whole_data = []
    for link in tqdm(links):
        data = parse_single_url(link)
        whole_data.append(data)

    df = pd.DataFrame(whole_data)
    df.to_csv(csv_path, index=False, header=['date', 'id', 'url', 'title', 'text'])


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    main()
