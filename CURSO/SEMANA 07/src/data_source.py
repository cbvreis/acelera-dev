import pandas as pd


class DataSource:
    def __init__(self):
        self.path_train = '../data/train.csv'
        self.path_test = '../data/test.csv'
        self.path_label = '../data/sample_submision.csv'

    def read_data(self, step_train=True) -> pd.DataFrame:
        '''
            Read data from data sources
        :param step_train: Boolean specifing if is train or test
        :return: pd.DataFrame with values and pd.Series with labels
        '''

        if step_train:
            return pd.read_csv(self.path_train)

        df = pd.read_csv(self.path_test)
        y = pd.read_csv(self.path_label)
        return df, y
