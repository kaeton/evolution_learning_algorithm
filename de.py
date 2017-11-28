import pandas as pd
import numpy as np
import sys


class DifferentialEvolution:
    def __init__(self,
                 range,
                 individual=30,
                 digit_setting=100,
                 seed=1234,
                 function_option=1,
                 learning_rate=0.4,
                 learning_iteration=50,
                 cr=0.5
                 ):
        np.random.seed(seed)
        self.range = range
        self.individual_number = individual
        self.digit_setting = digit_setting
        self.function_option = function_option
        self.function_calculater = CalculateFunction()
        self.feature_number = self.function_calculater.\
            feature_number_each_function(self.function_option)
        self.learning_rate = learning_rate
        self.cr_value = cr
        self.learning_iteration = learning_iteration
        self.dataset = self.initialize_vector()
        self.sort_by_function_value()

    # 初期値の設定
    def initialize_vector(self):
        low, high = [x * self.digit_setting for x in self.range]
        random_int = np.random.randint(
            low=low,
            high=high,
            size=(self.individual_number, self.feature_number)
        )
        random_indivisuals = np.divide(random_int, float(self.digit_setting))
        value_array = [self.function_calculater.calculate_function_value(x)
                       for x in random_indivisuals]
        dataset = np.concatenate(
            (
                random_indivisuals,
                np.array(value_array).reshape(len(value_array), 1)
            ),
            axis=1
        )

        input_columns = ["input_" + str(x) for x in range(self.feature_number)]
        columns = input_columns+['function_value']
        pandas_dataset = pd.DataFrame(dataset, columns=columns)

        return pandas_dataset

    def sort_by_function_value(self):
        self.dataset = self.dataset.sort_values(by=['function_value'])

    def evolve_training(self):
        record_number = len(self.dataset.index)
        for learning_iteration in range(self.learning_iteration):
            for i in range(int(record_number/2), record_number):
                while(1):
                    random_int = np.random.randint(
                        low=0,
                        high=int(record_number/2)-1,
                        size=3
                    )
                    if self.judge_same_number(random_int):
                        break

                evolution_seed = self.dataset.iloc[random_int]
                original_vector = self.dataset.iloc[i]
                new_individual = self.calculate_new_individual(
                    evolution_seed=evolution_seed,
                    original_vectors=original_vector
                )

                input_columns = ["input_" + str(x)
                                 for x in range(self.feature_number)]
                for x, column in enumerate(input_columns):
                    self.dataset[column].iloc[i] = new_individual[x]

                self.dataset["function_value"].iloc[i] = self.function_calculater.calculate_function_value(new_individual)

            self.sort_by_function_value()
            de_analyser.dataset.to_csv(
                "result_" + str(learning_iteration) + ".csv"
            )
            print(de_analyser.dataset["function_value"].iloc[0])

    def judge_same_number(self, array):
        for i in range(len(array)):
            if array[i] in array[i+1:]:
                return False

        return True

    # 今回のDEアルゴリズムの芯の部分
    def calculate_new_individual(self, evolution_seed, original_vectors):
        new_individual_input = []
        input_columns = ["input_" + str(x) for x in range(self.feature_number)]
        random_position = np.random.randint(
            low=0,
            high=self.feature_number,
            size=1
        )
        for feature in input_columns:
            rand3_vector = evolution_seed[feature]
            original_vector = original_vectors[feature]
            # step 1
            donor_vector = \
                rand3_vector.iloc[0] + \
                self.learning_rate * \
                (rand3_vector.iloc[1] - rand3_vector.iloc[2])
            # step 2
            random_value = np.random.randint(
                low=0,
                high=100,
                size=1
            )[0] / 100.0
            if (feature == random_position) or self.cr_value < random_value:
                new_individual_input.append(donor_vector)
            else:
                new_individual_input.append(original_vector)

        return new_individual_input


class CalculateFunction:
    def feature_number_each_function(self, function_option):
        self.function_option = function_option
        if self.function_option == 1:
            return 3
        elif self.function_option == 2:
            return 2

    def calculate_function_value(self, x):
        if self.function_option == 1:
            function_value = self.function_1(x)
        elif self.function_option == 2:
            function_value = self.function_2(x)
        return function_value

    def function_1(self, input_array):
        if len(input_array) != 3:
            sys.stderr.write("ERROR : Please input 3 feature array")
            exit(1)
        return_value = sum([x**2 for x in input_array])
        return return_value

    def function_2(self, input_array):
        if len(input_array) != 2:
            sys.stderr.write("ERROR : Please input 2 feature array")
            exit(1)
        x1, x2 = input_array
        return 100 * (x1 ** 2 - x2) ** 2 + (1 - x1) ** 2


if __name__ == "__main__":
    de_analyser = DifferentialEvolution(
        range=[-2.014, 2.014],
        # range=[-5.12, 5.12],
        individual=100,
        digit_setting=100,
        seed=1234,
        function_option=2,
        learning_rate=0.4,
        learning_iteration=50,
        cr=0.5
    )
    de_analyser.evolve_training()
