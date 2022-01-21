import numpy as np
import pandas as pd
from tabulate import tabulate

class NaiveBayesianClassifier:
    def __init__(self, k_folds, shuffle=False):
        self.k_folds = k_folds
        self.shuffle = shuffle

    def _separate_by_class(self):
        pass

    def _mean(self, data_poins: np.array):
        return np.mean(data_poins)

    def _stdev(self, data_poins: np.array):
        return np.std(data_poins)

    def summarize_dataset(self, dataset):
        header_names = ['Column name', 'Mean', "Stdev", 'Quantity']
        summary = self.summarize_by_class(dataset)
        print(tabulate(summary,
                       headers=header_names,
                       tablefmt='presto'))

    def summarize_by_class(self, dataset):
        """
        Two level pandas dataframe. First column level is class name,
        second describes mean, std, quantity of classes.
        """
        MEAN_str = 'mean'
        STDEV_str = 'stdev'
        QUANTITY_str = 'quantity'

        # create new, multi_index dataframe
        unique_classes = dataset.target.unique()
        unique_indecies = dataset.columns.to_list()

        columns = pd.MultiIndex.from_product([unique_classes,
                                              [MEAN_str, STDEV_str, QUANTITY_str]],
                                             names=['Class name', 'metric'])

        df_summary = pd.DataFrame(index=unique_indecies, columns=columns)

        for unique_class in list(unique_classes):

            mean = dataset[dataset.target == unique_class].mean().rename(MEAN_str)
            stdev = dataset[dataset.target == unique_class].std().rename(STDEV_str)
            quantity = dataset[dataset.target == unique_class].shape[0]

            # append data
            data = pd.DataFrame(
                {MEAN_str: mean,
                 STDEV_str: stdev,
                 QUANTITY_str: quantity}
            )

            df_summary[unique_class] = data

        return df_summary

    def split_kfold(self, dataset):
        for train_index in np.array_split(dataset, self.k_folds):
            yield train_index

    def algorithm(self, train, test):
        summarize = self.summarize_by_class(train)
        predictions = list()
        for row in test:
            output = self.predict(summarize, row)
            predictions.append(output)
        return(predictions)
