import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import code.config as config

import yfinance as yf
vix = yf.download("^VIX", start="1927-12-30", end="2023-12-29")
vix.to_csv(os.path.join(config.data_path,r'vix.csv'))

spy = yf.download("SPY", start="1927-12-30", end="2023-12-29")
spy.to_csv(os.path.join(config.data_path,r'spy.csv'))

irx = yf.download("^IRX", start="1927-12-30", end="2023-12-29")
irx.to_csv(os.path.join(config.data_path,r'irx.csv'))

tnx = yf.download("^TNX", start="1927-12-30", end="2023-12-29")
tnx.to_csv(os.path.join(config.data_path,r'tnx.csv'))

spx = yf.download("^GSPC", start="1927-12-30", end="2023-12-29")
spx.to_csv(os.path.join(config.data_path,r'spx.csv'))


from fredapi import Fred
import requests
import pandas as pd

api_key = os.getenv('fred_key')
fred = Fred(api_key)
def fetch_releases():
    """
    取得 FRED 大分类信息
    Args:
        api_key (str): 秘钥
    """
    r = requests.get('https://api.stlouisfed.org/fred/releases?api_key=' + api_key + '&file_type=json', verify=True)
    full_releases = r.json()['releases']
    full_releases = pd.DataFrame.from_dict(full_releases)
    full_releases = full_releases.set_index('id')
    full_releases.to_csv(os.path.join(config.data_path,'category_information_fred.csv'))
    return full_releases

def fetch_release_id_data(release_id):
    """
    Args:
        release_id (int): 大分类ID
    Returns:
        dataframe: 数据
    """
    econ_data = pd.DataFrame(index=pd.date_range(start='1928-01-01',end='2024-01-01', freq='MS'))
    series_df = fred.search_by_release(release_id, limit=3, order_by='popularity', sort_order='desc')
    for topic_label in series_df.index:
        econ_data[series_df.loc[topic_label].title] = fred.get_series(topic_label, observation_end='2024-01-01')
    return econ_data
full_releases = fetch_releases()
keywords = ["producer price", "consumer price", "fomc", "manufacturing", "employment"]
for search_keywords in keywords:
    search_result = full_releases.name[full_releases.name.apply(lambda x: search_keywords in x.lower())]
    econ_data = pd.DataFrame(index=pd.date_range(start='1928-01-01',end='2024-01-01', freq='MS'))

    for release_id in search_result.index:
        print("scraping release_id: ", release_id)
        econ_data = pd.concat([econ_data, fetch_release_id_data(release_id)], axis=1)
    econ_data.to_csv(os.path.join(config.data_path,f"{search_keywords}.csv"))
