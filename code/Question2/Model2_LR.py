import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import code.config as config

class LR:
    def __init__(self, y_id):
        '''
        :param y_id: choose from 4-12
        '''
        self.y_id = y_id

    def modify_dataset(self):
        factors = pd.read_csv(os.path.join(config.data_path, 'final_dataset.csv'), index_col=0)
        y_value = pd.read_csv(os.path.join(config.data_path, 'y_value.csv'), index_col=0)
        factors['Date'] = pd.to_datetime(factors['Date'])
        y_value['Date'] = pd.to_datetime(y_value['Date'], dayfirst=True)
        '''
        Date is the date to buy/sell, therefore factor data should move 1 day because if buy on 2nd, we can only know data <= 1st.
        We will use yesterday's factor data to predict today's SPX return
        '''
        y_date_modified = [np.nan] + y_value['Date'].to_list()[0:(y_value.shape[0] - 1)]
        y_value['Date'] = y_date_modified
        data = pd.merge(factors, y_value, on='Date').dropna()
        return data

    def split_dataset(self, data):
        factor_columns = ['spx', 'MA50', 'RSI', 'MACD_signal', 'vix', 'spx_volume', 'irx', 'tnx', 'erp', 'cpi',
                          'non_farm', 'GDP', 'PCE']
        scaler = StandardScaler()
        X = scaler.fit_transform(data[factor_columns])
        y = data[f'y_{self.y_id}'].to_list()
        date_column = data['Date']

        X_df = pd.DataFrame(X, columns=factor_columns)
        y_df = pd.DataFrame(y, columns=[f'y_{self.y_id}'])
        final_df = pd.concat([date_column, X_df, y_df], axis=1)
        train_df = final_df[final_df['Date'] < pd.to_datetime('2019-01-01')]
        test_df = final_df[final_df['Date'] >= pd.to_datetime('2019-01-01')]

        X_train = train_df[factor_columns].values
        y_train = train_df[f'y_{self.y_id}'].values

        X_test = test_df[factor_columns].values
        y_test = test_df[f'y_{self.y_id}'].values
        return X_train,X_test, y_train, y_test, test_df

    def train_linear_regression(self, X_train, X_test, y_train, y_test):
        linear_model = LinearRegression()
        linear_model.fit(X_train, y_train)
        y_pred_validation = linear_model.predict(X_test)
        mse_validation = mean_squared_error(y_test, y_pred_validation)
        r2_validation = r2_score(y_test, y_pred_validation)
        print(f"Validation MSE: {mse_validation}, R²: {r2_validation}")
        return linear_model, mse_validation, r2_validation

    def train_ridge_regression(self, X_train, X_test, y_train, y_test):
        ridge = Ridge()
        param_grid = {'alpha': [0.0001, 0.001, 0.1, 1, 10, 100]}
        ridge_cv = GridSearchCV(ridge, param_grid, cv=5)
        ridge_cv.fit(X_train, y_train)
        print(f"Best alpha: {ridge_cv.best_params_['alpha']}")
        y_pred_validation = ridge_cv.predict(X_test)
        mse_validation = mean_squared_error(y_test, y_pred_validation)
        r2_validation = r2_score(y_test, y_pred_validation)
        print(f"Validation MSE: {mse_validation}, R²: {r2_validation}")
        return ridge_cv.best_estimator_, mse_validation, r2_validation

    def predict_result(self,model,X_test, y_test, test_df):
        y_pred_test = model.predict(X_test)
        result_df = pd.DataFrame({
            'Date': test_df['Date'].values,
            'Real_Result': y_test,
            'Predict_Result': y_pred_test
        })
        print(result_df)
        return result_df


def test():
    results = []
    for y_id in range(4, 13):
        print(f"Running for y_id = {y_id}")
        lr = LR(y_id=y_id)
        data = lr.modify_dataset()
        X_train, X_test, y_train,y_test, test_df = lr.split_dataset(data)

        model_ridge, mse_ridge, r2_ridge = lr.train_ridge_regression(X_train, X_test, y_train, y_test)
        model_linear, mse_linear, r2_linear = lr.train_linear_regression(X_train, X_test, y_train, y_test)

        results.append({
            'y_id': y_id,
            'mse_ridge': mse_ridge,
            'r2_ridge': r2_ridge,
            'mse_linear': mse_linear,
            'r2_linear': r2_linear,
        })

        test_result_ridge = lr.predict_result(model_ridge, X_test, y_test, test_df)
        test_result_linear = lr.predict_result(model_linear, X_test, y_test, test_df)

        plt.figure(figsize=(10, 6))
        plt.plot(test_result_ridge['Date'], test_result_ridge['Real_Result'], label='Real Result', color='black',linewidth=2)
        plt.plot(test_result_ridge['Date'], test_result_ridge['Predict_Result'], label='Ridge Prediction', color='blue',linestyle='--')
        plt.plot(test_result_linear['Date'], test_result_linear['Predict_Result'], label='Linear Prediction',color='green', linestyle='--')
        plt.title(f'Prediction Results for y_id = {y_id}')
        plt.xlabel('Date')
        plt.ylabel('SPX Return')
        plt.legend()
        plt.grid(True)

        plt.savefig(os.path.join(config.result_path, f'Model2_LR/prediction_results_y_{y_id}.png'))
        plt.close()

    results_df = pd.DataFrame(results)
    return results_df

def main():
    '''
    :return: after analysing loss and figure, finally choose y_4 & linear regression
    '''
    lr = LR(y_id=4)
    data = lr.modify_dataset()
    X_train, X_test, y_train, y_test, test_df = lr.split_dataset(data)
    model_linear, mse_linear, r2_linear = lr.train_linear_regression(X_train, X_test, y_train, y_test)
    test_result_linear = lr.predict_result(model_linear, X_test, y_test, test_df)
    test_result_linear.to_csv(os.path.join(config.result_path, f'Model2_LR/predict_result.csv'))

if __name__ == "__main__":
    main()