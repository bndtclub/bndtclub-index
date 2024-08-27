import streamlit as st
import eurostat
import pandas as pd
import matplotlib.pyplot as plt
from data_processing import process_import_data, process_ICT_labour_import_data

st.write("Hello World!")

data = 'Index.xlsx'


#'''
# hold approach by getting data from excel. Now data direclty from eurostat API
#Employment_ICT = pd.read_excel(data, sheet_name='Employment_ICT', skiprows=11, usecols="B:N")
#ICT_labour_demand = pd.read_excel(data, sheet_name='ICT_labor_demand', skiprows=8, usecols="B:L")
#GVA_ICT_perc_of_total = pd.read_excel(data, sheet_name='ICT_GVA_perc_of_total', skiprows=11, usecols="B:M")
#'''

# Load the dataset
GVA_data_import = eurostat.get_data_df('namq_10_a10_e')
Employment_data_import = eurostat.get_data_df('namq_10_a10_e')
Labour_demand_ICT_data_import = eurostat.get_data_df('isoc_sk_oja1')

date_start = '2019Q4'
GVA_data = process_import_data(GVA_data_import, date_start)
Employment_data = process_import_data(Employment_data_import, date_start)
Labour_demand_ICT_data = process_ICT_labour_import_data(Labour_demand_ICT_data_import, date_start)

# Display the melted DataFrame to verify the structure
st.write("Melted DataFrame:")
#st.dataframe(GVA_data_import)
#st.dataframe(GVA_data_melted.sample(1000))
st.dataframe(GVA_data[(GVA_data['nace_r2'] == 'J')])
st.dataframe(Employment_data[(Employment_data['nace_r2']=='J') & (Employment_data['unit']=='PC_TOT_PER')])
st.dataframe(Labour_demand_ICT_data)

### Testing plotting functinality 

# Filter data for nace_r2 = 'J' and unit = 'PC_TOT_PER'
filtered_data = Employment_data[(Employment_data['nace_r2'] == 'J') & (Employment_data['unit'] == 'PC_TOT_PER') & (Employment_data['geo'] == 'IT')]

# Convert 'quarter' to datetime format for filtering
filtered_data['quarter'] = pd.PeriodIndex(filtered_data['quarter'], freq='Q').to_timestamp()
filtered_data['quarter'] = filtered_data['quarter'].dt.date  # Conversion to datetime.date

# Create a time range slider
min_time = filtered_data['quarter'].min()
max_time = filtered_data['quarter'].max()

selected_time_range = st.slider(
    "Select time range",
    min_value=min_time,
    max_value=max_time,
    value=(min_time, max_time)
)

# Filter data based on the selected time range
time_filtered_data = filtered_data[
    (filtered_data['quarter'] >= selected_time_range[0]) & 
    (filtered_data['quarter'] <= selected_time_range[1])
]

# Plot the data
st.write("Line Graph for Employment Data")
plt.figure(figsize=(10, 6))
plt.plot(time_filtered_data['quarter'], time_filtered_data['value'])
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Employment Data for nace_r2 = J and unit = PC_TOT_PER')
plt.xticks(rotation=45)
plt.grid(True)

# Display the plot
st.pyplot(plt)