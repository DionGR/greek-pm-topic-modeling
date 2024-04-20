import time
import pandas as pd
from urllib.request import urlopen
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from tqdm import tqdm


class WebScraper:
    """
    Code adapted from
    https://github.com/kritonp/primeminister-speeches-scrapper
    and completely re-written to be functioning and more modular, 
    as the original code was not working
    
    With permission from the original author:
    https://github.com/kritonp/primeminister-speeches-scrapper/issues/1
    """
    def __init__(self, base_url):
        self.base_url = base_url
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36"
        self.headless = True
        self.driver = None

    def init_driver(self):
        options = Options()
        options.add_argument(f"user-agent={self.user_agent}")

        if self.headless:
            options.add_argument("headless")

        return webdriver.Chrome(options=options)

    def fetch_article_links(self, url):
        n_scrolls = 200
        x_path_link = '//*[@id="td-outer-wrap"]//div[2]/h3/a'
        x_path_btn_pattern = '//*[@id="td-outer-wrap"]/div[3]/div/div/div[1]/div/div[{}]/a'

        self.driver.get(url)

        for _ in range(5):
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)

        for no in tqdm(range(n_scrolls), desc="Fetching Links"):
            try:
                button_xpath = x_path_btn_pattern.format(32 + no * 10)
                button = self.driver.find_element(By.XPATH, button_xpath)
                button.click()
                time.sleep(1.5)
            except NoSuchElementException:
                continue

        elements = self.driver.find_elements(By.XPATH, x_path_link)
        return [element.get_attribute('href') for element in elements]

    def fetch_article_contents(self, urls):
        return [self.parse_article(url) for url in tqdm(urls, desc="Parsing Articles")]

    @staticmethod
    def parse_article(url):
        def clean_text(text):
            text= text.replace("\n", " ").replace("\r", " ").replace("\t", " ").replace(",", " ") .replace(";", " ").replace("\"", " ").replace("'", " ").replace("\'", " ")
            return str(" ".join(text.split()))
                       
        try:
            page = urlopen(url)
            
            soup = BeautifulSoup(page, "html.parser")
            title = soup.title.string
            
            date = '-'.join(url.split("/")[-4:-1])
            text = " ".join(para.get_text() for para in soup.find_all("p"))
                        
            return {"date": clean_text(date), "id": clean_text(url.split("/")[-1]), "url": url, "title": clean_text(title), "text": clean_text(text)}
        except Exception as e:
            print(f"Error processing {url}: {e}")
            return None

    def run(self, category, data_path="", store=True):
        self.driver = self.init_driver()

        data_df = None
        file_path = data_path + f"data_{category}.csv"

        try:
            data_df = pd.read_csv(file_path)
            print(f"Cached data found for {category}, loading CSV...")
        except FileNotFoundError:
            category_url = self.base_url + category

            links = self.fetch_article_links(category_url)
            articles_data = self.fetch_article_contents(links)

            articles_data = [article for article in articles_data if article is not None]

            data_df = pd.DataFrame(articles_data)

            if store:
                print(f"Storing {category} data to {file_path}...")
                data_df.to_csv(file_path, index=False)
        except Exception as e:
            print(f"Error fetching data: {e}")
        finally:
            self.driver.quit()

        return data_df
