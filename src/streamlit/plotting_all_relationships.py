from data_loading_and_cleaning import DataLoadingAndCleaning
from plotly import express as px

class PlottingAllRelationships:
    
    def __init__(self, dlac):
        self.data = dlac.data
        self.all_relationships_fig = self.get_all_relationships_fig()
    
    def get_all_relationships_fig(self):
        fig = px.scatter_matrix(
            self.data,
            dimensions=list(self.data.columns),   # columns to include
            height=1080, width=1080
        )
        fig.update_traces(marker=dict(size=4, opacity=0.6))
        fig.update_layout(title="Scatter matrix (pairwise)")
        return fig