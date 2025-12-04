from data_loading_and_cleaning import DataLoadingAndCleaning
import matplotlib.pyplot as plt
import seaborn as sns

class CorrelationMatrix:
    def __init__(self, dlac: DataLoadingAndCleaning):
        self.data = dlac.data
        self.corr = self.data.corr()
        self.fig = self.make_correlation_matrix_figure()

    def make_correlation_matrix_figure(self):
        fig, ax = plt.subplots(figsize=(20, 20))
        sns.heatmap(self.corr, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title("Correlation Matrix")
        return fig