import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import code.config as config

class my_portfolio:
    def __init__(self,initial_assets):
        self.initial_assets = initial_assets
        self.spx_price = pd.read_csv(config.spx_price_path)
        self.spx_price.columns = ['Date', 'spx']
        self.y_value = pd.read_csv(os.path.join(config.data_path, 'y_value.csv'), index_col=0)
        self.y_value['Date'] = pd.to_datetime(self.y_value['Date'], errors='coerce')
        self.spx_price['Date'] = pd.to_datetime(self.spx_price['Date'], errors='coerce')
        self.model2_id = 4
        self.model1_id = 10

    def quantile_return(self, y_id, percentage):
        df_return = self.y_value[self.y_value['Date'] < '2019-01-01'][['Date',f'y_{y_id}']]
        len = df_return.shape[0]-y_id########for data in [len:] will use data in test dataset
        df_return = df_return.iloc[:len]
        if 0 <= percentage <= 1:
            buy_signal = df_return[f'y_{y_id}'].quantile(percentage)
        else:
            buy_signal = 'error percentage entered'
        return buy_signal

    def portfolio(self, model, percentage):
        '''
        :param model: should enter model1 or model2
        '''
        if model == 'model2':
            predict_result = pd.read_csv(os.path.join(config.result_path,'Model2_LR/predict_result.csv'), index_col=0)
            y_id = self.model2_id
        else:
            predict_result = pd.read_csv(os.path.join(config.result_path,'Model1_LSTM/predict_result.csv'), index_col=0)
            predict_result['Predicted Value'] =  predict_result['Predicted Value'].apply(lambda x: x[1:-2])
            y_id = self.model1_id
            predict_result = predict_result[predict_result['y_label'] == y_id][['Date','Predicted Value']]
            predict_result.columns = ['Date', 'Predict_Result']
            predict_result['Predict_Result'] = pd.to_numeric(predict_result['Predict_Result'], errors='coerce')
        buy_signal = self.quantile_return(y_id,percentage)
        print(f"Buy signal threshold: {buy_signal}")
        date_signal = predict_result[predict_result['Predict_Result'] >= buy_signal]['Date'].tolist()

        buy_sell_dates = []
        last_sell_date = None
        for signal_date in date_signal:
            signal_index = self.spx_price[self.spx_price['Date'] == signal_date].index
            signal_index = signal_index[0]

            if signal_index + 1 < self.spx_price.shape[0]:
                buy_date = self.spx_price.iloc[signal_index + 1]['Date']
            else:
                break

            if signal_index + (1+y_id) < self.spx_price.shape[0]:
                sell_date = self.spx_price.iloc[signal_index + 1+y_id]['Date']
            else:
                break

            if last_sell_date is None or buy_date > last_sell_date:
                buy_sell_dates.append((buy_date, sell_date))
                last_sell_date = sell_date

        return buy_sell_dates

    def portfolio_return(self, buy_sell_dates):
        trade_data = []
        total_assets = self.initial_assets
        first_buy_price = None
        win_count = 0

        for buy_date, sell_date in buy_sell_dates:
            buy_price = self.spx_price.loc[self.spx_price['Date'] == buy_date, 'spx'].values
            sell_price = self.spx_price.loc[self.spx_price['Date'] == sell_date, 'spx'].values

            if len(buy_price) > 0 and len(sell_price) > 0:
                buy_price = buy_price[0]
                sell_price = sell_price[0]

                if first_buy_price is None:
                    first_buy_price = buy_price

                shares_bought = total_assets / buy_price
                sell_value = shares_bought * sell_price
                profit = sell_value - total_assets
                if profit > 0:
                    win_count += 1
                trade_return = profit / total_assets
                total_assets = sell_value

                trade_data.append([buy_date, sell_date, profit, trade_return])
            else:
                print(f"Invalid buy/sell dates: {buy_date}, {sell_date}")

        trades_df = pd.DataFrame(trade_data, columns=['Buy_Date', 'Sell_Date', 'Profit', 'Return'])

        if first_buy_price is not None:
            total_return = (total_assets - self.initial_assets) / self.initial_assets
        else:
            total_return = None

        if len(trade_data) > 0:
            win_rate = win_count / len(trade_data)
        else:
            win_rate = None

        return trades_df, total_return, win_rate


def main():
    initial_assets = 100000
    portfolio = my_portfolio(initial_assets)
    # to make the transactions around 20 per year
    percentage1 = ((252/portfolio.model1_id)-18)/(252/portfolio.model1_id)
    percentage2 = ((252/portfolio.model2_id)-18)/(252/portfolio.model2_id)
    buy_sell_dates_model1 = portfolio.portfolio('model1', percentage1)
    buy_sell_dates_model2 = portfolio.portfolio('model2', percentage2)
    print(f'results for model1: ')
    trades_df_model1, total_return_model1, win_rate_model1 = portfolio.portfolio_return(buy_sell_dates_model1)
    print(trades_df_model1)
    print(f'total return: {total_return_model1} | win rate: {win_rate_model1}')
    print(f'results for model2: ')
    trades_df_model2,total_return_model2, win_rate_model2 = portfolio.portfolio_return(buy_sell_dates_model2)
    print(trades_df_model2)
    print(f'total return: {total_return_model2} | win rate: {win_rate_model2}')

if __name__ == "__main__":
    main()