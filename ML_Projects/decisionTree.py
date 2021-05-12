import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle


class Node:

    def __init__(self, dataset, depth, is_gini=True, purity_to_stop_split=0.95):
        self.right = None
        self.left = None
        self.feature_index = None
        self.value = None
        self.depth_in_tree = depth
        self.dataset = dataset
        self.label = None
        self.is_gini = is_gini
        self.purity_to_stop_split = purity_to_stop_split

    def split_one_node(self, index, value):
        right = self.dataset[self.dataset[:, index] <= value]
        left = self.dataset[self.dataset[:, index] > value]
        return left, right

    def get_features_to_check(self, num_columns):
        return range(num_columns - 1)

    def get_best_split(self):
        num_rows = self.dataset.shape[0]
        num_columns = self.dataset.shape[1]
        best_score = 999
        features_to_search = self.get_features_to_check(num_columns)
        for feature_index in features_to_search:
            for row_num in range(num_rows):
                left, right = self.split_one_node(feature_index, self.dataset[row_num, feature_index])
                score = self.calcualte_metrics(left, right)
                if score < best_score:
                    best_score = score
                    best_feature = feature_index
                    best_value = self.dataset[row_num, feature_index]
                    best_right = right
                    best_left = left
        return best_feature, best_value, best_right, best_left

    def calcualte_metrics(self, left_dataset, right_dataset, is_gini=True):
        metrics = 0
        total_size = len(left_dataset) + len(right_dataset)
        # total_size = left.shape[0] + right.shape[0]
        for branch in [left_dataset, right_dataset]:
            branch_size = len(branch)
            if branch_size == 0:
                continue
            # arr_branch = np.array(branch)
            labels = branch[:, -1]
            vals, freq = np.unique(labels, return_counts=True)
            probabilities = freq / branch_size
            gini_branch = 1 - (probabilities ** 2).sum()
            entropy_branch = (probabilities * np.log(probabilities)).sum()
            metrics_branch = gini_branch if is_gini else entropy_branch
            metrics += metrics_branch * branch_size / total_size
        return metrics

    def split_node_recursively(self, max_depth=6):
        purity = self.calc_node_purity()
        if self.depth_in_tree > max_depth or purity > self.purity_to_stop_split:
            self.label = self.calc_majority_label()
            return
        else:
            self.feature_index, self.value, right_dataset, left_dataset = self.get_best_split()
            self.right = Node(right_dataset, self.depth_in_tree + 1, self.is_gini)
            self.right.split_node_recursively(max_depth)
            self.left = Node(left_dataset, self.depth_in_tree + 1, self.is_gini)
            self.left.split_node_recursively(max_depth)

    def calc_majority_label(self):
        labels = self.dataset[:, -1]
        vals, freq = np.unique(labels, return_counts=True)
        max_index = np.argmax(freq)
        return vals[max_index]

    def calc_node_purity(self):
        labels = self.dataset[:, -1]
        vals, freq = np.unique(labels, return_counts=True)
        probabilities = freq / sum(freq)
        purity = np.max(probabilities)
        return purity

    def predict(self, row):
        if self.label != None:
            return self.label
        else:
            if row[self.feature_index] > self.value:
                return self.left.predict(row)
            else:
                return self.right.predict(row)

    def print_tree(self):
        if self.label != None:
            print('\t' * (self.depth_in_tree - 1), 'label is: ', self.label, 'and depth is: ', self.depth_in_tree)
        else:
            print('\t' * (self.depth_in_tree - 1), 'split is at: ', self.feature_index, ' ', self.value,
                  ' and depth is: ', self.depth_in_tree)
            # if self.right:
            self.right.print_tree()
            # if self.left:
            self.left.print_tree()

    def save_tree(self, file_name='my_tree'):
        pickle.dump(file_name, self)

    def load_tree(self, file_name='my_tree'):
        return pickle.load(file_name)

    def evalute(self, test_x, test_y):
        test_labels = [self.predict(row_data) for row_data in test_x]
        accuracy = 100 * (1 - np.sum(abs(test_labels - test_y) / len(y_test)))
        return accuracy


if __name__ == '__main__':
    # reading data and pre-processing
    file_name = r'datasets\wdbc.data.txt'
    data = pd.read_csv(file_name, header=None)

    y = data.iloc[:, 1]
    x = data.drop(1, axis=1)
    x = x.drop(0, axis=1)

    y_bool = y.apply(lambda x: 1 if x == 'M' else 0)

    x_train, x_test, y_train, y_test = train_test_split(x, y_bool, train_size=0.8)
    data = np.column_stack((x_train, y_train))
    data_test = np.column_stack((x_test, y_test))
    # 1
    root_node = Node(data, 1)
    root_node.split_node_recursively()
    root_node.print_tree()
    accuracy = root_node.evalute(x_test.values, y_test.values)

    # test_labels = []
    # for row_data in data_test:
    #     row_label = root_node.predict(row_data)
    #     test_labels.append(row_label)

    print('accuracy is {}'.format(accuracy))