from preprocessing import Preprocess
from sklearn.model_selection import train_test_split
from catboost import(
                    CatBoostRegressor,
                    Pool,
                    metrics, 
                    cv
) 
import argparse
import numpy as np
import pandas as pd

class Model:
    def __init__(self, path, lr, iters):
        self.path = path
        self.lr = lr
        self.iters = iters
        self.data = Preprocess(path)
        self.categoricals = self.data.categorical 
        self.df = self.data.preprocess()

        self.data_tmp, self.data_eval = train_test_split(self.df, test_size=0.01, 
                                                random_state=1
        ) 
        self.y_tmp = self.data_tmp['target_price']
        self.data_tmp.drop(columns = ['target_price'], inplace=True)

        self.data_train, data_test, y_train, y_test = train_test_split(
                                                self.data_tmp, self.y_tmp, 
                                                test_size=0.2, random_state=1
        )
        earlystop_params = {
        'od_type': 'Iter',
        'od_wait': 40
        }
        self.model = CatBoostRegressor(
            iterations=iters,
            learning_rate=lr,
            silent=True,
            **earlystop_params
        )

        self.train_pool = Pool(data=self.data_train,
                        label=y_train,
                        cat_features=self.categoricals
        )
        self.test_pool = Pool(data=data_test,
                        label=y_test,
                        cat_features=self.categoricals
        )

    def train_model(self):
        self.model.fit(self.train_pool, eval_set=self.test_pool, plot=True)


    def cv_model(self):
        params = self.model.get_params()
        cv_data = cv(
            Pool(self.data_tmp, self.y_tmp, cat_features=self.categoricals),
            params,fold_count=5, plot=True
        )
        print('Best validation RMSE score: {:.2f}±{:.2f} on step {}'.format(
        np.min(cv_data['test-RMSE-mean']),
        cv_data['test-RMSE-mean'][np.argmin(cv_data['test-RMSE-mean'])],
        np.argmin(cv_data['test-RMSE-mean'])
        ))
        

    def model_eval(self):
        feature_importances = self.model.get_feature_importance(self.train_pool)
        feature_names = self.data_train.columns
        for score, name in sorted(zip(feature_importances, feature_names), reverse=True):
            print('{}: {}'.format(name, score))
        
        eval_metrics = self.model.eval_metrics(self.test_pool,
                                             [metrics.MAE(),
                                             metrics.RMSE()
                                             ], plot=True
        )
    def model_test(self):
        seed = 1
        y = self.data_eval['target_price']
        X = self.data_eval.drop(['target_price'], axis=1)
        y = y.sample(1, random_state=seed)
        X = X.sample(1, random_state=seed)
        pred = self.model.predict(X)
        print(f'Predicted value: {pred}')
        print(f'Target value: {y.values}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Training Catboost Regression model'
        )
    parser.add_argument('--path',
                        type=str,
                        default=r'otomoto_price_prediction_data.csv',
                        help='Path to the data'
    )
    parser.add_argument('--lr',
                        type=float,
                        default=0.01,
                        help='Learning rate'
    )
    parser.add_argument('--iters',
                        type=int,
                        default=1000,
                        help='Number of iterations'
    )
    args = parser.parse_args()
    model = Model(
        path=args.path,
        lr=args.lr,
        iters=args.iters,
    )
    model.train_model()
    model.model_eval()
    model.cv_model()
    model.model_test()
