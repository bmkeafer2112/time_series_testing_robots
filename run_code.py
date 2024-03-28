# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 16:06:32 2023

@author: bmkea
"""

from denso import analysis, predicative_confidence_intervals, statistical_profile, cluster

# =============================================================================
# #Initial Parameters
# robot_name = 'denso_01'
# joint = 3
# signal = 'Torque'
# path = r'C:\Users\bmkea\Documents\Denso_Test_cell\Denso_Test_Cell_Data\experiments\control_run_10_31_2023.csv'
# start = '2023-10-31 09:00:00'
# end = '2023-10-31 12:00:00'
# anal = analysis(robot_name, joint, signal, path, start, end)
# =============================================================================

# =============================================================================
# #Analysis
# output1 = anal.data_input()
# anal.show_distribution(output1)
# output1 = anal.aggregate(output1, freq='1min', stat='mean')
# anal.ks_test(output1)
# anal.show_distribution(output1)
# output1 = anal.transforms(output1, transform='boxcox')
# anal.ks_test(output1)
# anal.show_distribution(output1)
# output1 = anal.transforms(output1, transform='zscore')
# anal.ks_test(output1)
# anal.show_distribution(output1)
# anal.chart_timeseries(output1)
# =============================================================================


# =============================================================================
# #ARIMA Model
# output1 = anal.data_input()
# output1 = anal.aggregate(output1, freq='1min', stat='mean')
# output1 = anal.transforms(output1, transform='boxcox')
# output1 = anal.transforms(output1, transform='zscore')
# anal.show_distribution(output1)
# anal.ks_test(output1)
# pci = predicative_confidence_intervals()
# #predictions = pci.auto_arima(output1, n_periods=120, freq='100ms', seasonal=True, seasonal_period=100)
# predictions = pci.auto_arima(output1, n_periods=120, freq='1min', seasonal=False, seasonal_period=42, force_seasonal=1)
# =============================================================================


# =============================================================================
# #Prophet
# output1 = anal.data_input()
# output1 = anal.aggregate(output1, freq='5min', stat='max')
# output1 = anal.transforms(output1, transform='boxcox')
# output1 = anal.transforms(output1, transform='zscore')
# pci = predicative_confidence_intervals()
# pci.and_prophet(output1, n_periods=15, freq='5min', seasonal=False)
# =============================================================================



# =============================================================================
# #Statistical Approach
# #Get Train Dataset for Anomaly Detection
# start = '2023-10-31 09:00:00'
# end = '2023-10-31 16:00:00'
# path = r'C:\Users\bmkea\Documents\Denso_Test_cell\Denso_Test_Cell_Data\experiments\control_run_10_31_2023.csv'
# robot_name = 'denso_01'
# attribute = 'Amp'
# sp= statistical_profile(path, robot_name)
# data = sp.data_input()
# 
# train_df = sp.transforms(dataframe=data, agg=True, freq='1min', stat='mean', trans=False)
# #train_df = sp.transforms(dataframe=data, agg=True, freq='1min', stat='mean', trans=True, trans_type='zscore')
# 
# #Get Test Dataset for anomaly detection
# start = '2023-11-02 11:00:00'
# end = '2023-11-02 12:00:00'
# path = r'C:\Users\bmkea\Documents\Denso_Test_cell\Denso_Test_Cell_Data\experiments\speed_run_11_02_2023.csv'
# robot_name = 'denso_01'
# attribute = 'Amp'
# sp= statistical_profile(path, robot_name, start, end)
# data = sp.data_input()
# 
# test_df = sp.transforms(dataframe=data, agg=True, freq='1min', stat='mean', trans=False)
# #test_df = sp.transforms(dataframe=data, agg=True, freq='1min', stat='mean', trans=True, trans_type='zscore')
# 
# single, anom = sp.anomaly_detection(train_df, test_df)
# =============================================================================

#Clustering
#Get Baseline Train Set
start = '2023-10-31 09:00:00'
end = '2023-10-31 16:00:00'
path = r'C:\Users\bmkea\Documents\Denso_Test_cell\Denso_Test_Cell_Data\experiments\control_run_10_31_2023.csv'
robot_name = 'denso_01'
clust = cluster(path, robot_name, start, end)
train_df = clust.data_input()
train_df = clust.transforms(dataframe=train_df, agg=True, freq='1min', stat='mean', trans=False)

#Get Test Set
start = '2023-11-02 11:00:00'
end = '2023-11-02 12:00:00'
path = r'C:\Users\bmkea\Documents\Denso_Test_cell\Denso_Test_Cell_Data\experiments\speed_run_11_02_2023.csv'
robot_name = 'denso_01'
clust = cluster(path, robot_name, start, end)
test_df = clust.data_input()
test_df = clust.transforms(dataframe=test_df, agg=True, freq='1min', stat='mean', trans=False)

data = clust.kmeans(train_df, test_df)








