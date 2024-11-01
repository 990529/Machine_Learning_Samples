import os
import sys
import pandas as pd
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_path = os.path.join(base_dir, 'data')
result_path = os.path.join(base_dir, 'result')
spx_price_path = os.path.join(data_path,r'SPX Index.csv')
spx_path = os.path.join(data_path,r'spx.csv')
vix_path = os.path.join(data_path,r'vix.csv')
irx_path = os.path.join(data_path,r'irx.csv')
tnx_path = os.path.join(data_path,r'tnx.csv')
consumer_path = os.path.join(data_path,r'consumer price.csv')
employment_path = os.path.join(data_path,r'employment.csv')
gdp_and_pce = os.path.join(data_path,r'gdp.csv')