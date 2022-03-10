import pandas as pd
import numpy as np
import copy


class Node:
    """Contains the information of the node and another nodes of the Decision Tree."""
    def __init__(self):
        self.feature = None
        self.next = None
        self.value = None
        self.children = None
        self.label = None


class ID3:
    """Decision Tree Classifier using ID3 algorithm."""
    def __init__(self):
        self.root_node = None
        self.label = None  # name of the label column

    def _get_feature_max_information_gain(self, df: pd.DataFrame, features: str) -> str:
        """Method finds feature that maximizes the information gain."""
        max_info_gain = -1
        max_info_feature = None

        for feature in features:  # iterate through features
            feature_info_gain = self._get_information_gain(feature, df)  # get information gain of the feature
            if max_info_gain < feature_info_gain:  # selecting feature name with highest information gain
                max_info_gain = feature_info_gain
                max_info_feature = feature

        return max_info_feature

    def _get_information_gain(self, feature_name: str, df_data: pd.DataFrame) -> float:
        """Calculates the information gain for a feature based on its entropy and system's entropy."""

        feature_value_list = np.unique(df_data[feature_name])  # unique values of the feature
        total_rows = df_data.shape[0]  # total number of features
        feature_info = 0.0

        for feature_value in feature_value_list:  # unique values in th feature
            feature_value_data = df_data[
                df_data[feature_name] == feature_value]  # filtering rows with that feature_value

            feature_value_count = feature_value_data.shape[0]

            feature_value_entropy = self.calculate_feature_entropy(feature_value_data)
            feature_info += feature_value_count / total_rows * feature_value_entropy

        return self._get_system_entropy(df_data) - feature_info  # calculating information gain by subtracting

    def calculate_feature_entropy(self, df: pd.DataFrame) -> float:
        class_count = df.shape[0]  # number of rows
        available_class = np.unique(df[self.label])

        feature_entropy = 0
        for single_class in available_class:
            label_class_count = df[df[self.label] == single_class].shape[0]  # row count of class c
            entropy_class = 0
            if label_class_count != 0:
                probability_class = label_class_count / class_count  # probability of the class
                entropy_class = probability_class * np.log(probability_class)  # entropy  # here canceled minus !
            feature_entropy -= entropy_class
        return feature_entropy

    def _get_system_entropy(self, df: pd.DataFrame) -> float:
        """Calculates the entropy of the system"""
        total_row = df.shape[0]  # the total size of the dataset
        class_list = np.unique(df[self.label])

        total_entropy = 0
        for single_class in class_list:  # for each class in the label
            class_entropy = df[df[self.label] == single_class].shape[0]  # number of the class
            class_entropy = (class_entropy / total_row) * np.log(class_entropy / total_row)
            total_entropy -= class_entropy

        return total_entropy

    def _id3(self, node, df, features):
        """Builds ID3 classifying tree recursively."""
        if not df.empty:  # if dataframe not empty

            if len(features) == 0:  # if there are no features to compute
                # return majority of the labels
                classes = list(df[self.label])
                node.label = max(classes, key=classes.count)
                return node

            if len(np.unique(df[self.label])) <= 1:  # if all samples are of the same class
                node.label = np.unique(df[self.label])[0]
                return node

            # choose feature with max information gain for given data
            max_info_feature = self._get_feature_max_information_gain(df, features)
            node.feature = max_info_feature  # assign the feature that spits the node

            node.children = []  # define children
            features.remove(max_info_feature)

            unique_feature_values = np.unique(df[max_info_feature])  # get unique values of the max info gain feature

            # df = df.drop(columns=[max_info_feature])

            # attribute value information gain
            # value_gain = {value: self._get_value_information_gain(value, df, max_info_feature) for value in unique_feature_values}
            # unique_feature_values = dict(sorted(value_gain.items(), key=lambda item: item[1], reverse=True))

            for value in unique_feature_values[::-1]:  # iterate through maximum information gain feature
                child_node = Node()
                child_node.value = value

                node.children.append(child_node)

                # max_feature_value =
                sub_data = copy.deepcopy(df)

                sub_data = sub_data[sub_data[max_info_feature] == value]
                sub_data = sub_data.drop(columns=[max_info_feature])

                # sub_data = df.where(df[max_info_feature] == value).dropna()  # reset_index(drop=True)


                # call the algorithm recursively
                recur = self._id3(child_node, sub_data, features)  # child_node.next = self._id3(child_node, sub_data)
            return node
        else:
            # if there are not samples left, return majority class in the dataset
            node.label = df[self.label].value_counts().idxmax()
            # create nodes under parent node
            return node

    def fit(self, df: pd.DataFrame, label):
        """
        Method used to build a classifying tree
        """

        self.root_node = Node()  # initiate root node
        self.label = label

        features = list(df.columns.drop(label))

        generated_tree = self._id3(self.root_node, copy.deepcopy(df), features)

        return generated_tree

    def make_prediction(self, series, node=None):
        if node is None:
            node = self.root_node

        # if node doesn't have any child node = it is a leaf node
        # return it's label
        if node.children is None:
            return node.label
        else:
            split_feature = node.feature  # get the splitting value of a root node

            for child in node.children:
                # decide which value the series has
                if series[split_feature] == child.value:
                    return self.make_prediction(series, child)
                else:
                    continue

    def predict(self, df_input):

        uniqs = list(df_input[self.label])
        most_common_label = max(uniqs, key=uniqs.count)

        df_output = copy.deepcopy(df_input)

        prediction_label = f'{self.label}_hat'

        df_output[prediction_label] = None  # here add self label

        for index, row in df_input.iterrows():
            pred_label = self.make_prediction(row)
            if pred_label is None:
                pred_label = most_common_label  # df[self.label].iloc[0]
            df_output.at[index, prediction_label] = pred_label

        return df_output


    def _get_value_information_gain(self, feature_value, df_data, max_inf_feature):
        total_row = df_data.shape[0]

        class_count = df_data.shape[0]  # number of rows
        feature_entropy = 0

        available_class = df_data[df_data[max_inf_feature] == feature_value][self.label].unique()

        for single_class in available_class:
            label_class_count = df_data[df_data[self.label] == single_class].shape[0]  # row count of class c
            entropy_class = 0
            if label_class_count != 0:
                probability_class = label_class_count / class_count  # probability of the class
                entropy_class = probability_class * np.log(probability_class)  # entropy  # here canceled minus !
            feature_entropy -= entropy_class

        feature_value_count = df_data[df_data[max_inf_feature] == feature_value].shape[0]
        feature_value_probability = feature_value_count / total_row

        feature_info = feature_value_probability * feature_entropy

        return feature_info
