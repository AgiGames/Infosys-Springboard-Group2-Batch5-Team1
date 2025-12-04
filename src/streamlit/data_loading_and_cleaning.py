import pandas as pd
import numpy as np

class DataLoadingAndCleaning:

    def __init__(self):
        self.data = pd.read_csv(r'data/data.csv')
        self.categorical_columns = []
        self.numerical_columns = list(set(self.data.columns) - set(self.categorical_columns))
        self.column_types = {
            'numerical_columns': self.numerical_columns,
            'categorical_columns': self.categorical_columns,
        }

        self.data = self.clean_data()

    def data_needs_cleaning(self):
        print(self.data.isna().sum())
        return True if self.data.isna().sum().sum() > 0 else False

    # define a function to clean our data
    def clean_data(self) -> pd.DataFrame:
        for column in self.data.columns:
            if column in self.column_types['numerical_columns']:
                self.data[column] = self.data[column].astype(np.float64)
                self.data[column] = self.data[column].fillna(self.data[column].mean())
            elif column in self.column_types['categorical_columns']:
                self.data[column] = self.data[column].fillna(self.data[column].mode()[0])
                self.data[column] = self.data[column].astype(str)
        
        return self.data
    
