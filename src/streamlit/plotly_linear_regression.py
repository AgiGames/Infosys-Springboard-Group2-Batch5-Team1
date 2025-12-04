from data_loading_and_cleaning import DataLoadingAndCleaning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import pandas as pd

class PlotlyLinearRegression():
    def __init__(self, dlac: DataLoadingAndCleaning, x_label, y_label):
        self.xlabel = x_label
        self.ylabel = y_label
        self.data = dlac.data
        self.x = self.data[['distance']]
        self.y = self.data[['time_elapsed']]
        self.prediction_results = self.train_predict()
        self.plotly_fig = self.plot_true_vs_pred()

    def train_predict(self):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.3)
        model = LinearRegression()
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        prediction_results = pd.DataFrame(
            {
                'x': x_test[self.xlabel].to_list(),
                'y_pred': pred.reshape(-1),
                'y_true': y_test[self.ylabel].to_list()
            }
        )
        return prediction_results

    def plot_true_vs_pred(self):
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=self.prediction_results['x'],
            y=self.prediction_results['y_pred'],
            mode='lines',
            name='Predicted'
        ))

        fig.add_trace(go.Scatter(
            x=self.prediction_results['x'],
            y=self.prediction_results['y_true'],
            mode='markers',
            name='True',
            opacity=0.3
        ))

        fig.update_layout(
            title="Predicted vs True (Distance-Time)",
            xaxis_title="Distance",
            yaxis_title="Time Elapsed"
        )

        return fig