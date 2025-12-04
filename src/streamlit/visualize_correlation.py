from data_loading_and_cleaning import DataLoadingAndCleaning
from correlation_matrix import CorrelationMatrix
from collections import defaultdict
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np

class VisualizeCorrelation:
    def __init__(self, dlac: DataLoadingAndCleaning, cm: CorrelationMatrix):
        self.data = dlac.data
        self.corr = cm.corr
        self.corrs_abs = self.get_corrs_abs()
        self.corrs_abs_plot = self.get_abs_corrs_plot()
        self.highly_corr_vars = self.get_highly_correlating_vars()
        self.highly_corr_vars_figs = self.get_highly_corr_vars_figs()

    def get_corrs_abs(self):
        correlations = []
        num_columns = len(self.data.columns)
        for i, column in enumerate(self.data.columns):
            for j in range(i + 1, num_columns):
                correlations.append(self.corr.loc[column, self.data.columns[j]])
        correlations = np.array(correlations).reshape(-1, 1)
        correlations_abs = np.abs(correlations)
        return correlations_abs

    def get_abs_corrs_plot(self):
        fig, ax = plt.subplots(figsize=(6, 4))
        plt.scatter(self.corrs_abs, [1] * len(self.corrs_abs))
        ax.set_title('Absolute Correlations Plot')
        ax.set_xlabel('Absolute Correlations')
        return fig
    
    def get_highly_correlating_vars(self):
        highly_corr_vars = defaultdict(list)
        num_columns = len(self.data.columns)
        for i, column in enumerate(self.data.columns):
            for j in range(i + 1, num_columns):
                ijth_correlation = self.corr.loc[column, self.data.columns[j]]
                if abs(ijth_correlation) > 0.2:
                    highly_corr_vars['var_1'].append(column)
                    highly_corr_vars['var_2'].append(self.data.columns[j])
                    highly_corr_vars['corr'].append(ijth_correlation)
        highly_corr_vars = pd.DataFrame(highly_corr_vars)
        return highly_corr_vars
    
    def get_highly_corr_vars_figs(self):
        figs = []
        num_highly_corr_vars = len(self.highly_corr_vars)
        for i in range(num_highly_corr_vars):
            var_1, var_2, corr = self.highly_corr_vars.iloc[i].var_1, self.highly_corr_vars.iloc[i].var_2, self.highly_corr_vars.iloc[i].corr
            
            fig = px.scatter(self.data, var_2, var_1)
            figs.append(fig)
        return figs