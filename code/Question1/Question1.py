import pandas as pd
import os
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import code.config as config

class Max_Return:
    def __init__(self,fee,transaction_limit,price):
        '''
        :param fee: fee is the transaction fee for scenario 1
        :param transaction_limit: transaction_limit is the limit for scenario 2
        :param price: price is the stock price list
        '''
        self.fee = fee
        self.transaction_limit = transaction_limit
        self.price = price

    #1. Scenario1
    def max_profit_with_fee(self):
        profit, track = self.search_scenario1()
        first_buy_idx = next((i for i, x in enumerate(track) if x == 1), None)
        first_buy_price = self.price[first_buy_idx] if first_buy_idx is not None else None
        profit_percentage = 100 * profit / ((1 + self.fee) * first_buy_price)
        return profit, profit_percentage

    def search_scenario1_simple(self):
        '''
        :param idx: the id from the first day to the last day
        :param hold_stock: hold stock(True) or not(False)
        '''
        n = len(self.price)
        hold, not_hold = [0] * (n + 1), [0] * (n + 1)
        hold[0] = -(1 + self.fee) * self.price[0]
        for i in range(1, n):
            hold[i] = max(hold[i - 1], not_hold[i - 1] - (1 + self.fee) * self.price[i])
            not_hold[i] = max(not_hold[i - 1], hold[i - 1] + (1 - self.fee) * self.price[i])
        return not_hold[n - 1]

    def search_scenario1(self):
        '''
        :param idx: the id from the first day to the last day
        :param hold_stock: hold stock(True) or not(False)
        **Use 1(buy), -1(sell), 0(keep) to track the trade**
        '''
        n = len(self.price)
        hold, not_hold = [0] * (n+1), [0] * (n+1)
        hold[0] = -(1 + self.fee) * self.price[0]
        hold_track, not_hold_track = [1],[0]
        for i in range(1, n):
            if not_hold[i-1] >= hold[i-1] + (1 - self.fee) * self.price[i]:
                not_hold[i] = not_hold[i-1]
                not_hold_track_new = not_hold_track+[0]#keep not hold
            else:
                not_hold[i] = hold[i-1] + (1 - self.fee) * self.price[i]
                not_hold_track_new = hold_track+[-1]#sale now
            if hold[i - 1] >= not_hold[i - 1] - (1 + self.fee) * self.price[i]:
                hold[i] = hold[i-1]
                hold_track_new = hold_track+[0]#keep hold
            else:
                hold[i] = not_hold[i - 1] - (1 + self.fee) * self.price[i]
                hold_track_new = not_hold_track + [1]#buy now
            not_hold_track = not_hold_track_new
            hold_track = hold_track_new
        print(len(not_hold_track))
        print(n)
        return not_hold[n-1], not_hold_track

    #2. Scenario2
    def max_profit_with_transaction_limits(self):
        profit_list, track_list = self.search_scenario2()
        profit = max(profit_list)
        k_value = profit_list.index(profit)
        first_buy_idx = next((i for i, x in enumerate(track_list[k_value]) if x == 1), None)
        first_buy_price = self.price[first_buy_idx] if first_buy_idx is not None else None
        profit_percentage = 100 * profit / ((1 + self.fee) * first_buy_price)
        return profit, profit_percentage

    def search_scenario2(self):
        '''
        :param transactions_left: transactions left for reaching the limit
        '''
        n = len(self.price)
        max_transactions = self.transaction_limit
        hold = [[-float('inf')] * (max_transactions + 1) for _ in range(n)]#n*k
        not_hold = [[0] * (max_transactions + 1) for _ in range(n)]#n*k
        hold_track = [[1] for _ in range(max_transactions + 1)]
        not_hold_track = [[0] for _ in range(max_transactions + 1)]

        for k in range(1, max_transactions + 1):
            hold[0][k] = -self.price[0]

        for i in range(1, n):
            not_hold_track[0] = not_hold_track[0] + [0]
            for k in range(1, max_transactions + 1):
                if not_hold[i - 1][k] > hold[i - 1][k] + self.price[i]:
                    not_hold[i][k] = not_hold[i - 1][k]
                    not_hold_track_new = not_hold_track[k] + [0]
                else:
                    not_hold[i][k] = hold[i - 1][k] + self.price[i]
                    not_hold_track_new = hold_track[k] + [-1]
                if hold[i - 1][k] > not_hold[i - 1][k - 1] - self.price[i]:
                    hold[i][k] = hold[i - 1][k]
                    hold_track_new = hold_track[k] + [0]
                else:
                    hold[i][k] = not_hold[i - 1][k - 1] - self.price[i]
                    hold_track_new = not_hold_track[k - 1] + [1]
                hold_track[k] = hold_track_new
                not_hold_track[k] = not_hold_track_new
        return not_hold[n - 1], not_hold_track


def main():
    spx_df = pd.read_csv(config.spx_price_path)
    spx_price = spx_df['PX_LAST'].to_list()#[100, 180, 260, 310, 40, 535, 695, 70,855,90, 80, 21, 100]##
    print(f'Check Data: {all(isinstance(item, (int, float)) for item in spx_price)}\n')
    max_return = Max_Return(fee = 0.02, transaction_limit = 2, price = spx_price)
    print("Scenario 1: Max profit with fee")
    c1 = time.time()
    profit_1, profit_percentage_1 = max_return.max_profit_with_fee()
    print(f"Profit: {profit_1:.2f} | Calculating time: {time.time()-c1}\n")

    print("Scenario 2: Max profit with transaction limits")
    c2 = time.time()
    profit_2, profit_percentage_2 = max_return.max_profit_with_transaction_limits()
    print(f"Profit: {profit_2:.2f} | Calculating time: {time.time()-c2}\n")

if __name__ == "__main__":
    main()




