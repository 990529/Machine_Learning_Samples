import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import code.config as config
import ta

def check_missing_and_inf_values(df):
    issues = {
        'NaN': [],
        'inf': [],
        '-inf': []
    }
    for index, row in df.iterrows():
        for col in df.columns:
            value = row[col]
            if pd.isna(value):
                issues['NaN'].append((index, col))
            elif value == np.inf:
                issues['inf'].append((index, col))
            elif value == -np.inf:
                issues['-inf'].append((index, col))
    return issues


def generate_technical_signals():
    '''
    1. Gnenerate technical signals
    '''
    spx = pd.read_csv(config.spx_path)
    spx['spx'] = spx['Adj Close']
    spx['MA50'] = spx['Adj Close'].rolling(window=50).mean()
    spx['RSI'] = ta.momentum.RSIIndicator(spx['Adj Close'], window=14).rsi()
    macd = ta.trend.MACD(spx['Adj Close'])
    spx['MACD'] = macd.macd()
    spx['MACD_signal'] = macd.macd_signal()
    technical_data = spx[['Date', 'spx', 'MA50', 'RSI', 'MACD_signal']]
    technical_data.to_csv(os.path.join(config.data_path, 'technical_data.csv'))
    return technical_data

def generate_market_data():
    '''
    1. VIX: start from 1990-01-02
    2. SPX VOLUMN: start from 1950-01-03
    3. 2-Year Treasury Yield: start from 1960-01-04
    4. 10-Year Treasury Yield: start from 1962-01-02
    5. Equity Risk Premium: start from 1962-01-02
    '''
    vix = pd.read_csv(config.vix_path)[['Date','Adj Close']].rename(columns={'Adj Close': 'vix'})

    spx = pd.read_csv(config.spx_path)[['Date','Volume']].rename(columns={'Volume': 'spx_volume'})
    spx_volume = spx[spx['spx_volume']>0]
    market_data = pd.merge(vix, spx_volume, on='Date')

    irx = pd.read_csv(config.irx_path)[['Date', 'Adj Close']].rename(columns={'Adj Close': 'irx'})
    market_data = pd.merge(market_data, irx, on='Date')

    tnx = pd.read_csv(config.tnx_path)[['Date', 'Adj Close']].rename(columns={'Adj Close': 'tnx'})
    market_data = pd.merge(market_data, tnx, on='Date')

    spx_return = pd.read_csv(config.spx_path)[['Date','Adj Close']].rename(columns={'Adj Close': 'spx'})
    spx_return['spx Annual Return'] = spx_return['spx'].pct_change(periods=252)
    merge_spx_tnx = pd.merge(spx_return, tnx, on='Date')
    merge_spx_tnx['tnx percentage'] = merge_spx_tnx['tnx']/100
    merge_spx_tnx['erp'] = merge_spx_tnx['spx Annual Return']-merge_spx_tnx['tnx percentage']
    erp = merge_spx_tnx[['Date','erp']]
    market_data = pd.merge(market_data, erp, on='Date')
    issues = check_missing_and_inf_values(market_data)
    print(f'check market data: {issues}')
    return market_data

def generate_macro_data():
    '''
    1. cpi: start from 1947-01-01, monthly
    2. non_farm: start from 1939-01-01, monthly
    3. gdp: start from 1947-01-01, quarter
    4. pce: start from 1947-01-01, quarter
    '''
    consumer = pd.read_csv(config.consumer_path, header=0)
    consumer.rename(columns={consumer.columns[0]: 'Date',consumer.columns[1]:'cpi'}, inplace=True)
    cpi = consumer[consumer['cpi'].notna()][['Date','cpi']]

    employment = pd.read_csv(config.employment_path, header=0)
    employment.rename(columns={employment.columns[0]: 'Date', employment.columns[6]: 'non_farm'}, inplace=True)
    non_farm = employment[employment['non_farm'].notna()][['Date', 'non_farm']]
    macro_data = pd.merge(cpi, non_farm, on='Date')

    df = pd.read_csv(config.gdp_and_pce)
    quarter_month_map = {
        'Q1': [1, 2, 3],
        'Q2': [4, 5, 6],
        'Q3': [7, 8, 9],
        'Q4': [10, 11, 12]
    }
    rows = []
    for index, row in df.iterrows():
        year = row['Year']
        quarter = row['Quarter']
        gdp = row['GDP']
        pce = row['PCE']
        months = quarter_month_map[quarter]
        for month in months:
            date = pd.Timestamp(year=year, month=month, day=1)
            rows.append({
                'Date': date,
                'GDP': gdp,
                'PCE': pce
            })
    gdp_and_pce = pd.DataFrame(rows)
    macro_data['Date'] = pd.to_datetime(macro_data['Date'])
    macro_data = pd.merge(macro_data, gdp_and_pce, on='Date')
    issues = check_missing_and_inf_values(macro_data)
    print(f'check macro data: {issues}')
    return macro_data

def generate_y_value():
    spx = pd.read_csv(config.spx_price_path).rename(columns={'date':'Date','PX_LAST': 'SPX'})
    period = [4,5,6,7,8,9,10,11,12]
    spx.set_index('Date', inplace=True)
    for i in period:
        spx[f'SPX_{i}days_later'] = spx['SPX'].shift(-i)
        spx[f'y_{i}'] = (spx[f'SPX_{i}days_later'] - spx['SPX']) / spx['SPX']
    y_value = spx[[f'y_{i}' for i in period]].reset_index()
    return y_value

def main():
    technical_data = generate_technical_signals()
    technical_data['Date'] = pd.to_datetime(technical_data['Date'])
    print('Successfully generate technical signals for SPX')

    market_data = generate_market_data()
    market_data['Date'] = pd.to_datetime(market_data['Date'])
    print('Successfully generate market data')
    final_dataset = pd.merge(technical_data, market_data, on='Date')

    macro_data = generate_macro_data()
    print('Successfully generate macro data')
    macro_data.set_index('Date', inplace=True)
    macro_data_daily = macro_data.resample('D').ffill()
    final_dataset = pd.merge(final_dataset, macro_data_daily, on='Date')

    y_value = generate_y_value()
    print('Successfully generate y_value')
    final_dataset.to_csv(os.path.join(config.data_path,r'final_dataset.csv'))
    y_value.to_csv(os.path.join(config.data_path,r'y_value.csv'))

if __name__ == "__main__":
    main()