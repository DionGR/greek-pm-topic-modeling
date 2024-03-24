from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from tqdm import tqdm

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


def scrapping(fetch_url, speeches_url_log_file):
    opts = Options()
    opts.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36")
    opts.add_argument("headless")
    n = 63  # No. of scrolls

    # x paths for Selenium
    x_path_link = '//*[@id="td-outer-wrap"]//div[2]/h3/a'
    x_path_btn = '//*[@id="td-outer-wrap"]/div[3]/div/div/div[1]/div/div[{}]/a'

    driver = webdriver.Chrome(options=opts)
    driver.get(fetch_url)

    # scroll down page n times
    for i in range(0, 5):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)

    for no in tqdm(range(0, n)):
        try:
            button = driver.find_element("xpath", x_path_btn.format(32 + no * 10))
            
            try:
                button.click()
                time.sleep(1.5)
            except NoSuchElementException:
                time.sleep(10)
                button.click()
                time.sleep(5)
        except NoSuchElementException:
            continue

    sel_links = driver.find_elements("xpath", x_path_link)

    links = []
    for link in sel_links:
        links.append(link.get_attribute('href'))
    driver.quit()

    # Create logging file with links
    with open(speeches_url_log_file, 'w') as fp:
        for item in links:
            fp.write("%s\n" % item)

    return links

import requests
from urllib.request import urlopen
from bs4 import BeautifulSoup


def get_year(speech_url):
    return speech_url.split("/")[-4]


def get_month(speech_url):
    return speech_url.split("/")[-3]


def get_day(speech_url):
    return speech_url.split("/")[-2]


def get_id(speech_url):
    return speech_url.split("/")[-1]


def create_date(speech_url):
    """
    :param speech_url: fetch url
    :return: YYYY-MM-DD
    """
    return str(get_year(speech_url)+'-'+get_month(speech_url)+"-"+get_day(speech_url))


def get_archive_urls(url):
    """
    :param url: fetch url
    :return: Retrieve archive urls per month
    """
    page = urlopen(url)
    html = page.read().decode("utf-8")
    soup = BeautifulSoup(html, "html.parser")
    options_list = soup.findAll('option')
    archive_url = []
    for option in options_list:
        archive_url.append(option['value'])
    return archive_url


def parse_single_url(speech_url):
    page = urlopen(speech_url)
    html = page.read().decode("utf-8")
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string
    id = get_id(speech_url)
    date = create_date(speech_url)

    # getting all the paragraphs
    text = ''
    for para in soup.find_all("p"):
        text = text + ' ' + para.get_text()

    data = [date, id, speech_url, title, text]
    return data


def read_log_file(filename):
    with open(filename) as f:
        links = f.read().splitlines()
    return links


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