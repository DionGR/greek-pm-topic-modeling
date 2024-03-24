from typing import List
from utils.web_scraper import WebScraper
import pandas as pd

class GreekPMDataloader:
    BASE_URL = "https://primeminister.gr/category/activity/"
    CATEGORIES = ["statements", "speeches", "letters", "articles", "addresses", "op-eds-interviews",  "press-releases", "press-conferences"]
    DATA_PATH = "../data/"
    
    def __init__(self):
        self.scraper = WebScraper(self.BASE_URL)
    
    def load_categories(self, *categories: List[str]):
        category_dfs = {}
        
        for category in categories:
            if category not in self.CATEGORIES:
                raise ValueError(f"Invalid category: {category}")
            
            category_url = self.BASE_URL + category
            articles_data = self.scraper.run(category_url, data_path=self.DATA_PATH)
            
            category_dfs[category] = articles_data
            
        return category_dfs
                
    def get_available_categories(self):
        return self.CATEGORIES
