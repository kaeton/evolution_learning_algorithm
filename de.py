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
                 function_option=1,
                 dropout=15,
                 learning_rate=0.4,
                 cr=0.5
                 ):
        # parameter setting
        # random.seed(seed)
        self.feature_number = feature_number
        self.range = range
        self.individual_number = individual
        self.digit_setting = digit_setting
        self.function_option = function_option
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.cr_value=cr
        self.dataset = self.initialize_vector(seed)
        self.sort_by_function_value()
        self.dataset.to_csv("hoge.csv")

    # 初期値の設定
    def initialize_vector(self, seed):
        # TODO: seed設定うまくいってないっぽい
        random.seed(seed)
        low, high = [x * self.digit_setting for x in self.range]
        random_int = np.random.randint(low=low,
                                       high=high,
                                       size=(self.individual_number, self.feature_number)
                                       )
        # decimal_array = [int_val for int_val in [x for x in random_int]]
        decimal_array = self.decimal_list_maker(random_int)
        random_indivisuals = np.divide(random_int, float(self.digit_setting))
        # if self.function_option == 1:
        #     value_array = [function_1(x) for x in random_indivisuals]
        # else:
        #     value_array = [function_2(x) for x in random_indivisuals]
        value_array = [self.calculate_function_value(x) for x in random_indivisuals]

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
        for i in range(self.feature_number):
            pandas_dataset[decimal_columns[i]] = decimal_array.T[i]

        return pandas_dataset

    def calculate_function_value(self, x):
        if self.function_option == 1:
            function_value = function_1(x)
        else:
            function_value = function_1(x)
        return function_value

    def sort_by_function_value(self):
        self.dataset = self.dataset.sort_values(by=['function_value'])

    # indivisualの配列を受け取り、それぞれの値を二進数に変換して返す
    def decimal_list_maker(self, int_np_array):
        decimal_1darray = []
        for i in range(np.shape(int_np_array)[0]):
            indivisual = []
            for j in range(np.shape(int_np_array)[1]):
                indivisual.append(self.decimal_maker(int_np_array[i][j]))

            decimal_1darray.append(indivisual)

        decimal_array = np.array(decimal_1darray)
        return decimal_array

    def decimal_maker(self, x):
        decimal_value = format(x, '010b')
        return decimal_value.replace('-', '1')

    def evolve_training(self):
        print(self.dataset.shape)
        record_number = len(self.dataset.index)
        # print(range(record_number/))
        # TODO : record numberの上限値設定
        for i in range(int(record_number/2), record_number):
            random_int = np.random.randint(low=0,
                                           high=int(record_number/2)-1,
                                           size=3
                                           )
            print(random_int)
            #TODO :  duplicate 判定
            evolution_seed = [self.dataset.iloc[random_selection]
                              for random_selection in random_int]
            evolution_seed = self.dataset.iloc[random_int]
            original_vector = self.dataset.iloc[i]
            new_individual = self.calculate_new_individual(evolution_seed=evolution_seed,
                                                           original_vectors=original_vector)
            print("evolution_seed")
            print(evolution_seed)
            print("original_vector")
            print(original_vector)
            print("new_individual")
            print(new_individual)

            print("break")
            # for i in range(3):


            # print(self.dataset.iloc[i])

    # 今回のDEアルゴリズムの芯の部分
    def calculate_new_individual(self, evolution_seed, original_vectors):
        print(evolution_seed)
        new_individual_input = []
        input_columns = ["input_" + str(x) for x in range(self.feature_number)]
        # TODO : ここ全体やばい、二重ループ不要
        for feature in input_columns:
            rand3_vector = evolution_seed[feature]
            original_vector = original_vectors[feature]


            print(original_vector)
            # step 1
            donor_vector = rand3_vector.iloc[0] + self.learning_rate * (rand3_vector.iloc[1] - rand3_vector.iloc[2])
            print(donor_vector)
            donor_vector = round(donor_vector, 2)
            print(donor_vector)
            random_position = np.random.randint(low=0, high=self.feature_number, size=1)
            # step 2
            for i in range(self.feature_number):
                random_value = np.random.randint(low=0, high=100, size=1)[0]/100.0
                if (i == random_position) or self.cr_value < random_value:
                    new_individual_input.append(donor_vector)
                else:
                    new_individual_input.append(original_vector)

        print(new_individual_input)

        return new_individual_input










        function_value_new_individual = self.calculate_function_value(new_individual_input)
        print(function_value_new_individual)




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
        function_option=1,
        learning_rate=0.4,
        cr=0.5
    )
    # print(function_1([2,2,2]))
    # print(function_2([2,2]))

    de_analyser.evolve_training()
