import pandas as pd


class Preprocess:
    def __init__(self, file_path='otomoto_price_prediction_data.csv'):
        self.df = pd.read_csv(file_path)
        self.categorical = ['make', 'model','gearbox',
        'fuel_type', 'damaged', 'is_business'
        ]
        self.numeric = ['vehicle_year', 'mileage',
         'engine_capacity', 'engine_power', 'target_price'
        ]

    def remove_outlier_IQR(self, df:pd.DataFrame)->pd.DataFrame:
        """Remove outliers by applying IQR method.

        Args:
            df (pd.DataFrame): DataFrame containing numeric values.

        Returns:
            pd.DataFrame: DataFrame without outliers.
        """
        Q1=df.quantile(0.25)
        Q3=df.quantile(0.75)
        IQR=Q3-Q1
        df_final=df[~((df<(Q1-1.5*IQR)) | (df>(Q3+1.5*IQR)))]
        return df_final
    
    def preprocess(self)->pd.DataFrame:
        """Remove duplicates, NaNs and outliers.

        Returns:
            df_final (pd.DataFrame): Returns preprocessed DataFrame.
        """
        self.df.dropna(inplace=True)
        self.df.drop_duplicates(inplace=True)

        return self.df

    @property
    def categoricals(self):
        return self.categorical
