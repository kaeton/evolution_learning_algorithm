import pandas as pd
import numpy as np
import random
from decimal import Decimal
import sys
import re

class DifferentialEvolution:
    def __init__(self,
                 feature_number=2,
                 range=[-5.11, 5.11],
                 individual=30,
                 digit_setting=100,
                 seed=1234,
                 function_option=1
                 ):
        # parameter setting
        # random.seed(seed)
        self.feature_number = feature_number
        self.range = range
        self.individual_number = individual
        self.digit_setting = digit_setting
        self.function_option = function_option
        self.dataset = self.initialize_vector(seed)
        self.sort_by_function_value()
        print(self.dataset)
        print(type(self.dataset["function_value"][0]))
        print(type(self.dataset["input_1"][0]))
        self.dataset.to_csv("hoge.csv")

    # 初期値の設定
    def initialize_vector(self, seed):
        random.seed(seed)
        low, high = [x * self.digit_setting for x in self.range]
        random_int = np.random.randint(low=low,
                                       high=high,
                                       size=(self.individual_number, self.feature_number)
                                       )
        # decimal_array = [int_val for int_val in [x for x in random_int]]
        decimal_array = self.decimal_list_maker(random_int)
        random_indivisuals = np.divide(random_int, float(self.digit_setting))
        if self.function_option == 1:
            value_array = [function_1(x) for x in random_indivisuals]
        else:
            value_array = [function_2(x) for x in random_indivisuals]

        dataset = np.concatenate(
            (
                random_indivisuals,
                np.array(value_array).reshape(len(value_array), 1)
            ),
            axis=1
        )

        input_columns = ["input_" + str(x) for x in range(self.feature_number)]
        decimal_columns = ["decimal_" + str(x) for x in range(self.feature_number)]
        columns = input_columns+['function_value']
        pandas_dataset = pd.DataFrame(dataset, columns=columns)
        print(decimal_array.T[1])
        for i in range(self.feature_number):
            pandas_dataset[decimal_columns[i]] = decimal_array.T[i]

        print(pandas_dataset)
        return pandas_dataset

    def sort_by_function_value(self):
        self.dataset = self.dataset.sort_values(by=['function_value'])

    # ランダムに作成したindivisualの配列を受け取り、それぞれの値を二進数に変換して返す
    def decimal_list_maker(self, int_np_array):
        decimal_1darray = []
        for i in range(np.shape(int_np_array)[0]):
            indivisual = []
            for j in range(np.shape(int_np_array)[1]):
                decimal_value = format(int_np_array[i][j], '010b')
                cut_minus_bin_value = decimal_value.replace('-', '1')
                indivisual.append(cut_minus_bin_value)

            decimal_1darray.append(indivisual)

        decimal_array = np.array(decimal_1darray)
        return decimal_array

    def evolve_training(self):
        evolve

    # def calculate_function_value(self):
        # write something

    # def evolution(self):
        # lskalks


def function_1(input_array):
    if len(input_array) != 3:
        sys.stderr.write("ERROR : Please input 3 feature array")
        exit(1)
    return_value = sum([x**2 for x in input_array])
    return np.round(return_value, 5)
    # return [x for x in return_value]

def function_2(input_array):
    x1, x2 = input_array
    return 100 * (x1 ** 2 - x2) ** 2 + (1 - x1) ** 2


if __name__ == "__main__":
    de_analyser = DifferentialEvolution(
        feature_number=3,
        range=[-5.11,5.11],
        individual=30,
        digit_setting=100,
        seed=1234,
        function_option=1
    )
    # print(function_1([2,2,2]))
    # print(function_2([2,2]))

    de_analyser.evolve_training()
