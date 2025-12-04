from data_loading_and_cleaning import DataLoadingAndCleaning
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import pandas as pd
import numpy as np

class Clustering:
    def __init__(self, dlac: DataLoadingAndCleaning, xlabel, ylabel):
        self.data = dlac.data
        xy = []
        for i in range(len(self.data)):
            xy.append((self.data[xlabel].iloc[i], self.data[ylabel].iloc[i]))
        xy = np.array(xy)

        xy_scaled = StandardScaler().fit_transform(xy)
        km = KMeans(n_clusters=3, init='k-means++', random_state=123, n_init=50)
        clusters = km.fit_predict(xy_scaled)

        xy_plot_data = pd.DataFrame(
            {
                'x': xy_scaled[..., 0],
                'y': xy_scaled[..., 1]
            }
        )

        self.xy_cluster_plot = px.scatter(xy_plot_data, 'x', 'y', color=clusters, labels={"x": xlabel, "y": ylabel})