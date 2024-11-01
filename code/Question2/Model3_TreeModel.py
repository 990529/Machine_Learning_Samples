from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas.core.common import random_state
from sklearn.datasets import load_diabetes
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import code.config as config
import warnings
warnings.filterwarnings("ignore")
class XGB:
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
        df = data[['Date']+factor_columns+[f'y_{self.y_id}']]
        print(df)
        train_df = df[df['Date'] < pd.to_datetime('2019-01-01')]
        test_df = df[df['Date'] >= pd.to_datetime('2019-01-01')]

        X_train = train_df[factor_columns].values
        y_train = train_df[f'y_{self.y_id}'].values
        X_test = test_df[factor_columns].values
        y_test = test_df[f'y_{self.y_id}'].values
        return X_train, X_test, y_train, y_test, test_df

    def train_decisiontree(self, X_train, X_test, y_train, y_test, test_df):
        param_dist = {
            'criterion': ['mse', 'friedman_mse'],
            'max_depth': randint(5, 30),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(5, 20),
            'max_features': ['auto', 'sqrt']
        }
        regressor = DecisionTreeRegressor(random_state=50)
        random_search = RandomizedSearchCV(estimator=regressor, param_distributions=param_dist, n_iter=100, cv=5,
                                           scoring='neg_mean_squared_error')
        #randomizedSearchCV 不像 GridSearchCV 那样穷举所有可能的超参数组合，而是从用户定义的参数分布中 随机选择 一定数量的参数组合，接着对每个组合进行评估。
        random_search.fit(X_train, y_train)
        # 输出最优参数和对应的评分
        print("最优参数：", random_search.best_params_)
        print("最优评分：", -random_search.best_score_)
        # 使用最优参数重新训练模型
        best_regressor = random_search.best_estimator_
        best_regressor.fit(X_train, y_train)
        plt.figure(figsize=(20, 10))
        feature_names = ['spx', 'MA50', 'RSI', 'MACD_signal', 'vix', 'spx_volume', 'irx', 'tnx', 'erp', 'cpi',
                          'non_farm', 'GDP', 'PCE']
        plot_tree(best_regressor, filled=True, feature_names=feature_names)
        plt.draw()
        plt.show()

        # 预测测试集结果
        y_pred = best_regressor.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print('预测数据的均方误差（MSE）为： {:.4f}'.format(mse))
        rmse = mean_squared_error(y_test, y_pred, squared=False)  # 直接计算 RMSE
        print('预测数据的均方根误差（RMSE）为： {:.4f}'.format(rmse))
        mae = mean_absolute_error(y_test, y_pred)
        print('预测数据的平均绝对误差（MAE）为： {:.4f}'.format(mae))
        r2 = r2_score(y_test, y_pred)
        print('预测数据的 R² 值为： {:.4f}'.format(r2))
        result_df = pd.DataFrame({
            'Date': test_df['Date'].values,
            'Real_Result': y_test,
            'Predict_Result': y_pred
        })
        return result_df

    def train_randomforest(self, X_train, X_test, y_train, y_test, test_df):
        from sklearn.ensemble import RandomForestRegressor as RFR
        from sklearn.model_selection import cross_validate, KFold, GridSearchCV
        param_grid_simple = {"criterion": ["squared_error", "poisson"],
                             'n_estimators': [*range(20, 100, 5)],
                             'max_depth': [*range(10, 25, 2)],
                             "max_features": ["log2", "sqrt", 16, 32, 64, "auto"],
                             "min_impurity_decrease": [*np.arange(0, 5, 10)]
                             }
        # 定义回归器
        reg = RFR(random_state=83)
        # 进行5折交叉验证
        cv = KFold(n_splits=5, shuffle=True, random_state=83)
        reg = RFR(random_state=1412, verbose=True, n_jobs=-1)
        cv = KFold(n_splits=5, shuffle=True, random_state=83)
        search = GridSearchCV(estimator=reg,
                              param_grid=param_grid_simple,
                              scoring="neg_mean_squared_error",
                              verbose=True,
                              cv=cv,
                              n_jobs=-1)
        search.fit(X_train, y_train)
        # 输出最优参数和对应的评分
        print("最优参数：", search.best_params_)
        print("最优评分：", -search.best_score_)
        # 使用最优参数重新训练模型
        best_parameter = search.best_estimator_
        best_parameter.fit(X_train, y_train)

        y_pred = best_parameter.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print('预测数据的均方误差（MSE）为： {:.4f}'.format(mse))
        rmse = mean_squared_error(y_test, y_pred, squared=False)  # 直接计算 RMSE
        print('预测数据的均方根误差（RMSE）为： {:.4f}'.format(rmse))
        mae = mean_absolute_error(y_test, y_pred)
        print('预测数据的平均绝对误差（MAE）为： {:.4f}'.format(mae))
        r2 = r2_score(y_test, y_pred)
        print('预测数据的 R² 值为： {:.4f}'.format(r2))
        result_df = pd.DataFrame({
            'Date': test_df['Date'].values,
            'Real_Result': y_test,
            'Predict_Result': y_pred
        })
        return result_df

    def train_xgboost(self, X_train, X_test, y_train, y_test, test_df):
        xgb_model = xgb.XGBRegressor()
        GS = GridSearchCV(xgb_model, {"colsample_bytree":[1.0],"min_child_weight":[1.0,1.2]
                            ,'max_depth': [3,4,6], 'n_estimators': [500,1000]}, verbose=1)
        GS.fit(X_train, y_train)
        print(GS.best_params_)

        eval_set = [(X_train, y_train), (X_test, y_test)]
        gbm = xgb.XGBRegressor(**GS.best_params_)
        gbm.fit(X_train, y_train, eval_set=eval_set, verbose=True)
        y_pred = gbm.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print('预测数据的均方误差（MSE）为： {:.4f}'.format(mse))

        rmse = mean_squared_error(y_test, y_pred, squared=False)  # 直接计算 RMSE
        print('预测数据的均方根误差（RMSE）为： {:.4f}'.format(rmse))

        mae = mean_absolute_error(y_test, y_pred)
        print('预测数据的平均绝对误差（MAE）为： {:.4f}'.format(mae))

        r2 = r2_score(y_test, y_pred)
        print('预测数据的 R² 值为： {:.4f}'.format(r2))
        result_df = pd.DataFrame({
            'Date': test_df['Date'].values,
            'Real_Result': y_test,
            'Predict_Result': y_pred
        })
        return result_df

def main():
    '''
    :return: after analysing loss and figure, finally choose y_4 & linear regression
    '''
    xgb = XGB(y_id=4)
    data = xgb.modify_dataset()
    X_train, X_test, y_train, y_test, test_df = xgb.split_dataset(data)

    #result_decisiontree = xgb.train_decisiontree(X_train, X_test, y_train, y_test, test_df)
    #result_decisiontree.to_csv(os.path.join(config.result_path, f'Model3_XGB/predict_result_decisiontree.csv'))

    result_randomforest = xgb.train_randomforest(X_train, X_test, y_train, y_test, test_df)
    result_randomforest.to_csv(os.path.join(config.result_path, f'Model3_XGB/predict_result_randomforest.csv'))

    #result_xgb = xgb.train_xgboost(X_train, X_test, y_train, y_test, test_df)
    #result_xgb.to_csv(os.path.join(config.result_path, f'Model3_XGB/predict_result.csv'))

if __name__ == "__main__":
    main()



