from plotting_all_relationships import PlottingAllRelationships
from data_loading_and_cleaning import DataLoadingAndCleaning
from plotly_linear_regression import PlotlyLinearRegression
from visualize_correlation import VisualizeCorrelation
from anomaly_detection import AnomalyDetection
from correlation_matrix import CorrelationMatrix
from clustering import Clustering
import streamlit as st

dlac = DataLoadingAndCleaning()
cm = CorrelationMatrix(dlac)
vc = VisualizeCorrelation(dlac, cm)
plr = PlotlyLinearRegression(dlac, 'distance', 'time_elapsed')
par = PlottingAllRelationships(dlac)
c = Clustering(dlac, 'time_elapsed', 'total_calories')
ad = AnomalyDetection(dlac)

page_container = st.empty()

def load_page(title: str):
    with page_container.container():
        st.title(title, text_alignment='justify')
        if title == 'Data Loading and Cleaning':
            st.markdown("### Cleaned Data")
            st.dataframe(dlac.data)
            code = """
def clean_data(self) -> pd.DataFrame:
    for column in self.data.columns:
        if column in self.column_types['numerical_columns']:
            self.data[column] = self.data[column].astype(np.float64)
            self.data[column] = self.data[column].fillna(self.data[column].mean())
        elif column in self.column_types['categorical_columns']:
            self.data[column] = self.data[column].fillna(self.data[column].mode()[0])
            self.data[column] = self.data[column].astype(str)
    
    return self.data
            """
            st.code(code, language='python')
        
        elif title == "Correlation Matrix":
            st.pyplot(cm.fig)

        elif title == 'Plotting Correlations and Visualising Highly Correlating Variables':
            st.markdown("> Plotting our absolute correlations to see which variables with which correlations can we plot.")
            st.pyplot(vc.corrs_abs_plot)
            st.markdown("***We see that variables with absolute correlation greater than ```0.2``` is apt for plotting.***")
            st.text("\n")
            st.markdown("## Visualizing the highly correlating variables.")
            for fig in vc.highly_corr_vars_figs:
                st.plotly_chart(fig)
        
        elif title == 'Fitting Regression Line (Distance vs Time Elapsed)':
            st.plotly_chart(plr.plotly_fig)

        elif title == 'Plotting all Relationships (Scatter Matrix)':
            st.plotly_chart(par.all_relationships_fig)
        
        elif title == 'Clustering (Time Elapsed vs Total Calories)':
            st.plotly_chart(c.xy_cluster_plot)

        elif title == 'Anomaly Detection':
            st.markdown("### ***Given some data, this detector says if it significantly deviates from the norm.***")
            heart_rate = st.number_input("Heart Rate")
            time_elapsed = st.number_input("Time Elapsed During Run")
            average_running_cadence = st.number_input("Average Running Cadence")
            total_calories = st.number_input("Total Calories Burned")
            distance = st.number_input("Distance Covered During Run")
            submit_data = st.button("Submit Data")

            if submit_data:
                x_star = [heart_rate, time_elapsed, average_running_cadence, total_calories, distance]
                anomaly = ad.flag_anomaly(x_star)
                if anomaly:
                    st.warning("The given data is an anomaly.")
                else:
                    st.success("The given data is not an anomaly.")

st.sidebar.title("Fit Pulse Data Analysis and Anomaly Detection")

data_cleaning_button = st.sidebar.selectbox(
    "Choose Subtopic",
    [
        'Data Loading and Cleaning',
        'Correlation Matrix',
        'Plotting Correlations and Visualising Highly Correlating Variables',
        'Fitting Regression Line (Distance vs Time Elapsed)',
        'Plotting all Relationships (Scatter Matrix)',
        'Clustering (Time Elapsed vs Total Calories)',
        'Anomaly Detection',
    ]
)

if data_cleaning_button:
    page_container.empty()          # wipe main area only
    load_page(data_cleaning_button) # load new page into container