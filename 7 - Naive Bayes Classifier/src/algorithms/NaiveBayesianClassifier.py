import numpy as np
import pandas as pd
from tabulate import tabulate
from scipy.stats import norm
from sklearn.model_selection import train_test_split


class NaiveBayesianClassifier:
    def __init__(self, test_size):
        self.test_size = test_size

        self.MEAN_str = 'mean'
        self.STDEV_str = 'stdev'
        self.QUANTITY_str = 'quantity'

    @staticmethod
    def _mean(self, data_poins: np.array):
        return np.mean(data_poins)

    @staticmethod
    def _stdev(self, data_poins: np.array):
        return np.std(data_poins)

    @staticmethod
    def calculate_gaussian_probability(x, mean, stdev) -> float:
        """
        Calculates probability of input x in the gaussian distribution
        with given mean and stdev parameters.
        """
        return norm(mean, stdev).pdf(x)

    @staticmethod
    def split_train_test(dataset, test_size, shuffle=True):
        return train_test_split(dataset, test_size=test_size, shuffle=shuffle)

    def summarize_dataset(self, dataset):
        summary = self.summarize_by_class(dataset)
        return summary

    @staticmethod
    def print_summary(summary):
        header_names = ['Attributes'] + list(summary.columns)
        print(tabulate(summary,
                       headers=header_names,
                       tablefmt='presto'))

    def summarize_by_class(self, dataset):
        """
        Two level pandas dataframe. First column level is class name,
        second describes mean, std, quantity of classes
        """

        # create new, multi_index dataframe
        unique_classes = dataset.target.unique()

        # dataset.drop(columns=['target'], inplace=True)# drop target columns

        unique_indices = dataset.columns.to_list()[:-1]

        columns = pd.MultiIndex.from_product([unique_classes,
                                              [self.MEAN_str, self.STDEV_str, self.QUANTITY_str]],
                                             names=['Class name', 'metric'])

        df_summary = pd.DataFrame(index=unique_indices, columns=columns)

        for unique_class in list(unique_classes):

            mean = dataset[dataset.target == unique_class].mean().rename(self.MEAN_str)
            stdev = dataset[dataset.target == unique_class].std().rename(self.STDEV_str)
            quantity = dataset[dataset.target == unique_class].shape[0]

            # append data
            data = pd.DataFrame(
                {self.MEAN_str: mean.iloc[:-1],
                 self.STDEV_str: stdev.iloc[:-1],
                 self.QUANTITY_str: quantity}
            )

            df_summary[unique_class] = data

        return df_summary

    def calculate_class_probabilities(self, summaries: pd.DataFrame, data_sample: pd.Series):
        """
        Calculate the probabilities of predicting each class for a given row.
        """
        classes = list(summaries.columns.levels[0])  # get available classes
        attribute_quantity = list(summaries.index)

        total_rows = sum([summaries.loc[:, classes[class_name]]['quantity'][0] for class_name in classes])

        probabilities = dict()
        for class_name in classes:  # iterate through available classes

            class_df = summaries.loc[:, classes[class_name]]  # slice dataframe containing info regarding onw class

            class_sample_quantity = class_df['quantity'][0]
            probabilities[class_name] = class_sample_quantity / total_rows  # probability of a class

            for attribute in attribute_quantity:
                data_row = class_df.loc[attribute]
                sample_attribute_value = data_sample[attribute]

                mean = data_row[self.MEAN_str]
                stdev = data_row[self.STDEV_str]

                probabilities[class_name] *= self.calculate_gaussian_probability(sample_attribute_value, mean, stdev)

        return probabilities

    def predict(self, summaries, row):
        """
        Predict class for given row sample.
        """
        probabilities = self.calculate_class_probabilities(summaries, row)
        best_label, best_prob = None, -1
        for class_value, probability in probabilities.items():
            if best_label is None or probability > best_prob:
                best_prob = probability
                best_label = class_value
        return best_label

    @staticmethod
    def accuracy_metric(actual, predicted):
        print(predicted)
        print(actual)
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0

    def algorithm(self, train, test):
        summarize = self.summarize_by_class(train)
        predictions = list()
        for row_idx in range(len(test.index)):
            output = self.predict(summarize, test.iloc[row_idx])
            predictions.append(output)
        return predictions

    def fit_predict(self, dataset, shuffle=True):
        train_df, test_df = self.split_train_test(dataset, test_size=self.test_size, shuffle=shuffle)

        predicted_values = self.algorithm(train_df, test_df)
        actual_values = list(test_df.target)

        return pd.Series(actual_values), pd.Series(predicted_values)
