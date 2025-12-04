from data_loading_and_cleaning import DataLoadingAndCleaning
import pandas as pd
import numpy as np

class AnomalyDetection:

    def __init__(self, dlac: DataLoadingAndCleaning):
        self.data = dlac.data
        x_data = self.data.loc[:, ['heart_rate', 'time_elapsed', 'avg_running_cadence', 'total_calories']]
        y_data = self.data.loc[:, ['distance']]
        self.data_numpy = np.hstack([x_data.to_numpy(), y_data.to_numpy()])

        self.mu = np.mean(self.data_numpy, axis=0 )
        self.cov = np.cov(self.data_numpy, rowvar=False)

    def flag_anomaly(self, x_star):
        x_star = np.array(x_star)
        inv_cov = np.linalg.inv(self.cov)
        d = np.sqrt((x_star - self.mu).T @ inv_cov @ (x_star - self.mu))
        if d >= 3:
            print("The given data is an anomaly.")
            return True
        return False