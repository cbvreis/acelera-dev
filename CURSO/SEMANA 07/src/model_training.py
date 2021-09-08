from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from catboost import CatBoostRegressor

from joblib import dump, load

from data_source import DataSource
from preprocessing import Preprocessing
from experiments import Experiments


class ModelTraining:
    def __init__(self):
        self.data = DataSource()
        self.preprocessing = None
        self.algorithms = {'linear': LinearRegression(),
                                  'ridge': Ridge(),
                                  'decision_tree': DecisionTreeRegressor(),
                                  'random_forest': RandomForestRegressor(),
                                  'svm': SVR(),
                                  'catboost': CatBoostRegressor()}

    def model_training(self,model_name):
        '''
            Train the model.
        :return:  Dicti with trained model, preprocessing used and columns used in training
        '''
        pre = Preprocessing()
        print('Loading Data')
        df = self.data.read_data(step_train=True)
        print('Training preprocessing')
        X_train, y_train = pre.process(df, step_train= True)
        print('Training model')

        model_obj = self.algorithms[model_name]
        model_obj.fit(X_train,y_train)
        model = {
                    'model_obj': model_obj,
                    'preprocessing': pre,
                    'columns' : pre.feature_names,
        }
        print(model)
        dump(model, '../output/model.pkl')
        return model

