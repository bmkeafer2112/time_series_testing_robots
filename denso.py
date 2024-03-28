# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 13:25:46 2023

@author: bmkea
"""

import pandas as pd
import scipy.stats as ss
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import pmdarima as pm
from prophet import Prophet
from sklearn.cluster import KMeans


class analysis():
    """
    This class is to analyze, print charts, process, and transform timeseries data
    """
    
    def __init__(self, robot_name, path, joint, signal, start='None', end='None'):
        print ('transforming data')
        self.robot_name = robot_name
        self.joint = joint
        self.signal = signal
        self.attribute = signal + "_" + str(joint)
        self.path = path
        self.start = start
        self.end = end

    def data_input(self):
        """
        This is to read in the dataframe from a CSV, elimnate unused columns, convert datetime,
        select desired attribute, and filter for a single robot
        """
        denso_data = pd.read_csv(self.path)
        denso_data = denso_data[['Time_Stamp', 'Robot_Name', self.attribute]]
        denso_data.query(f'Robot_Name == "{self.robot_name}"', inplace=True)
        denso_data = denso_data.drop(columns=['Robot_Name'])
        
        #If start and end time provided, slice the data accordingly
        if self.start != 'None':
            denso_data = denso_data[(denso_data['Time_Stamp'] > self.start) & (denso_data['Time_Stamp'] < self.end)]
        
        denso_data['Time_Stamp'] = pd.to_datetime(denso_data['Time_Stamp'])
        denso_data = denso_data.set_index('Time_Stamp')
        
        return denso_data
    
    def aggregate(self, dataframe='None', freq='1min', stat='max'):
        """
        This is to aggregate the raw data and downsample by the specified rate,
        this returns mean and max aggregations, defaults to using the parameters
        from class if a dataframe is not provided
        """
        #Check for dataframe or use CSV path in default parameters
        if type(dataframe) == str:
            denso_data = self.data_input()
        else:
            denso_data = dataframe
            
        #Resample for mean and max values
        if stat == 'max':
            denso_data = denso_data.resample(freq).max()
        elif stat == 'mean':
            denso_data = denso_data.resample(freq).mean()
        
        return denso_data
    
    def transforms(self, dataframe='None', transform='zscore'):
        """
        This is to transform/scale data. Choose zscore, minmax, or boxcox
        Defaults to using the parameters from class if a dataframe is not provided
        """
        #Check for dataframe or use CSV path in default parameters
        if type(dataframe) == str:
            denso_data = self.data_input()
        else:
            denso_data = dataframe
        
        #Z-score Standardization
        if transform == 'zscore':
            denso_data = denso_data.apply(ss.zscore)

        
        #MinMax Standardization    
        elif transform =='minmax':
            trans = MinMaxScaler()
            array = trans.fit_transform(denso_data)
            denso_data = pd.DataFrame(array, columns=denso_data.columns, index=denso_data.index)
        
        #BoxCox Standardization
        elif transform =='boxcox':
            #Shift data since boxcox cannot transform negative data
            if denso_data.min().values[0] < 0:
                denso_data = denso_data + ((denso_data.min().values[0] * -1) + 0.0001)
            array, lam = ss.boxcox(denso_data[denso_data.columns[0]])
            denso_data = pd.DataFrame(array, columns=denso_data.columns, index=denso_data.index)
            print('Lambda: %f' % lam)
            
        return denso_data
        
    def ks_test(self, dataframe='None', dist="norm"):
        """
        This will compare the distribution of an attribute to a specified standard
        distribution using a Kolmogorov–Smirnov test. the default is a normal(parametric) comparison. 
        Will use the first column of a dataframe.
        Defaults to using the parameters from class if a dataframe is not provided
        """
        #Check for dataframe or use CSV path in default parameters
        if type(dataframe) == str:
            denso_data = self.data_input()
        else:
            denso_data = dataframe
        
        #Apply test to first column of dataframe and printing result
        ser = denso_data[denso_data.columns[0]]   
        x = ss.kstest(ser, dist)
        print(x)
    
    def show_distribution(self, dataframe='None', bins=100):
        """
        This creates a histogram of the attribute to display the distribution of the attributes
        Defaults to using the parameters from class if a dataframe is not provided
        """
        
        #Check for dataframe or use CSV path in default parameters
        if type(dataframe) == str:
            denso_data = self.data_input()
        else:
            denso_data = dataframe
        #Graph distriubtion    
        denso_data.plot.hist(bins=bins, alpha=0.5)
        
    def chart_timeseries(self, dataframe='None'):
        """
        This simply charts the dataframe on a line chart
        """
        #Check for dataframe or use CSV path in default parameters
        if type(dataframe) == str:
            denso_data = self.data_input()
        else:
            denso_data = dataframe
        plt.cla()
        ser = denso_data[denso_data.columns[0]]
        plt.figure(figsize=(10, 6))
        ser.plot()
    
    
    def get_best_distribution(data):
        """
        This compares the data attribute to common distributions and returns the best fit,
        it uses a goodness of fit test to accomplish this
        """
        print('this might be broken')
        dist_names = ["norm", "exponweib", "weibull_max", "weibull_min", "pareto", "genextreme"]
        dist_results = []
        params = {}
        for dist_name in dist_names:
            dist = getattr(ss, dist_name)
            param = dist.fit(data)
    
            params[dist_name] = param
            # Applying the Kolmogorov-Smirnov test
            D, p = ss.kstest(data, dist_name, args=param)
            print("p value for "+dist_name+" = "+str(p))
            dist_results.append((dist_name, p))
    
        # select the best fitted distribution
        best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
        # store the name of the best fit and its p value
        
    
        print("Best fitting distribution: "+str(best_dist))
        print("Best p value: "+ str(best_p))
        print("Parameters for the best fit: "+ str(params[best_dist]))
    
        return best_dist, best_p, params[best_dist]

class predicative_confidence_intervals():
    
    def __init__(self):
        print('working on predictions')
        
    def auto_arima(self, timeseries, n_periods, freq, seasonal=False, seasonal_period=1, force_seasonal=None):
        """
        This trains an ARIMA (Auto-Regressive Integrated Moving Average) model and makes future predictions.
        There are several parameters that can be set but p,q, and m will have the greatest impact. This will
        automatically us the model with smallest AIC score (after testing multiple different combinations of parameters)
        to make predictions
        """
        automodel = pm.auto_arima(timeseries, 
                          start_P=1,
                          max_P=1,
                          start_Q=1,
                          max_Q=1,
                          D=force_seasonal,
                          test='kpss',
                          seasonal=True,
                          m=seasonal_period,
                          trace=True)
        # Forecast
        fc, confint = automodel.predict(n_periods=n_periods, 
                                        return_conf_int=True, alpha=0.05)
        # Create index
        fc_ind = pd.date_range(timeseries.index[timeseries.shape[0]-1], 
                               periods=n_periods, freq=freq)
        # Forecast series
        fc_series = pd.Series(data=fc.values, index=fc_ind.values)
        
        #Set Intervals
        lower_series = pd.Series(confint[:, 0], index=fc_ind)
        upper_series = pd.Series(confint[:, 1], index=fc_ind)
       
        #Create Plot
        plt.figure(figsize=(10, 6))
        plt.plot(timeseries)
        plt.plot(fc_series, color="red")
        plt.fill_between(lower_series.index, 
                         lower_series, 
                         upper_series, 
                         color="k", 
                         alpha=0.25)
        plt.legend(("past", "forecast", "95% confidence interval"),  
                   loc="upper left")
        plt.show()
        return fc, fc_series, fc_ind
    
    def and_prophet(self, dataframe, n_periods, freq, seasonal=True, seasonal_periods_in_days=1):
        """
        This uses META's Prophet api to build a predicative model. This uses confidence intervals,
        any data points outside of these intervals should be considered anomalies. This model 
        takes trend, season, and 'special days' into account. There are several parameters that
        need to be tuned to make it successful"""
        df = dataframe
        df['ds'] = df.index.values
        df.reset_index()
        #rename columns to ds, y
        #df.drop(df.columns[0], axis=1, inplace=True)
        df.rename(columns={df.columns[0]: "y"}, inplace = True)
        if seasonal==True:
            #Change seasonality to what you actually want it to be
            m = Prophet(weekly_seasonality=False)
            m.add_seasonality(name='seasonality', period=seasonal_periods_in_days, fourier_order=5)
        else:
            #Simply remove weekly seasonality
            m = Prophet(weekly_seasonality=False)      
        m.fit(df)
        future = m.make_future_dataframe(periods=n_periods, freq=freq)
        future.tail()
        forecast = m.predict(future)
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
        fig1 = m.plot(forecast)
        fig2 = m.plot_components(forecast)
        
class statistical_profile():
    
    def __init__(self, path, robot_name, start='None', end='None', attribute='Amp'):
        self.path = path
        self.robot_name = robot_name
        self.start = start
        self.end = end
        self.attribute = attribute
        
        print('finding anomalies')
        
        
    def data_input(self):
        """
        This is to read in the dataframe from a CSV, elimnate unused columns, convert datetime,
        select desired attribute, and filter for a single robot
        """
        denso_data = pd.read_csv(self.path)
        denso_data.query(f'Robot_Name == "{self.robot_name}"', inplace=True)
        denso_data = denso_data.drop(columns=['Robot_Name'])
        
        #Create Dataframe to keep timestamps
        time = pd.DataFrame()
        time['time'] = denso_data['Time_Stamp']
        
        #Filter dataframe to only keep attributes we are interest in
        denso_data = denso_data.filter(regex=self.attribute)
        denso_data['Time_Stamp'] = time['time']
        
        #If start and end time provided, slice the data accordingly
        if self.start != 'None':
            denso_data = denso_data[(denso_data['Time_Stamp'] > self.start) & (denso_data['Time_Stamp'] < self.end)]
        
        denso_data['Time_Stamp'] = pd.to_datetime(denso_data['Time_Stamp'])
        denso_data = denso_data.set_index('Time_Stamp')
        
        return denso_data
        
    def transforms(self, dataframe, agg=True, freq='1min', stat='mean', trans=True, trans_type='zscore'):
        """
        This is to transform/scale data. Choose zscore, minmax, or boxcox
        Defaults to using the parameters from class if a dataframe is not provided
        """
        
        denso_data = dataframe
        
        if agg == True:
            #Resample for mean and max values
            if stat == 'max':
                denso_data = denso_data.resample(freq).max()
            elif stat == 'mean':
                denso_data = denso_data.resample(freq).mean()
                
        if trans == True:
                
            #Z-score Standardization
            if trans_type == 'zscore':
                denso_data = denso_data.apply(ss.zscore)

            
            #MinMax Standardization    
            elif trans_type =='minmax':
                trans = MinMaxScaler()
                array = trans.fit_transform(denso_data)
                denso_data = pd.DataFrame(array, columns=denso_data.columns, index=denso_data.index)
            
# =============================================================================
#             #BoxCox Standardization
#             elif trans_type =='boxcox':
#                 #Shift data since boxcox cannot transform negative data
#                 if denso_data.min().values[0] < 0:
#                     denso_data = denso_data + ((denso_data.min().values[0] * -1) + 0.0001)
#                 array, lam = ss.boxcox(denso_data[denso_data.columns].values)
#                 denso_data = pd.DataFrame(array, columns=denso_data.columns, index=denso_data.index)
#                 print('Lambda: %f' % lam)
# =============================================================================
                     
        return denso_data
        
              
    def anomaly_detection(self, train_df, test_df):
        all_df = train_df.append(test_df)
        all_df = self.transforms(all_df, agg=False, trans=True, trans_type='zscore')
        for column in all_df:           
            #anomalies = all_df[(all_df[column] > 3).any(axis=1)]
            single_att = all_df[column]
            single_att = single_att.reset_index(drop=True)
            single_att = single_att.to_frame()
            #single_att['index1'] = single_att.index
            col_name = single_att.columns
            #col_name = col_name[0]
            #anomalies = single_att.query('@col_name > 3')
            anomalies = single_att[(single_att.iloc[:, 0] > 3) | (single_att.iloc[:, 0] < -3)]
            anomaly_count = len(anomalies)
            
            
            plt.style.use('fivethirtyeight')
            plt.title(label=str(col_name[0]) % col_name, fontsize=16, color="black") 
            plt.scatter(y=single_att.iloc[:, 0], x=single_att.index.values, color='green', marker='.', label='good')
            plt.scatter(y=anomalies.iloc[:, 0], x=anomalies.index.values, color = 'red', marker='*', label='anomaly')
            plt.text(0.5, 0.5,'Anomaly Count: %i' % anomaly_count, fontsize = 12, horizontalalignment='center', verticalalignment='top')
            plt.legend(loc="lower left")
            plt.show()

            
        return single_att, anomalies
    
class cluster():
    
    def __init__(self, path, robot_name, start='None', end='None'):
        self.path = path
        self.robot_name = robot_name        
        self.start = start
        self.end = end
        
    def data_input(self):
        """
        This is to read in the dataframe from a CSV, elimnate unused columns, convert datetime,
        select desired attribute, and filter for a single robot
        """
        denso_data = pd.read_csv(self.path)
        denso_data.query(f'Robot_Name == "{self.robot_name}"', inplace=True)
        denso_data = denso_data.drop(columns=['Robot_Name'])
        
        #Create Dataframe to keep timestamps
        time = pd.DataFrame()
        time['time'] = denso_data['Time_Stamp']
        
        #Filter dataframe to only keep attributes we are interest in
        denso_data = denso_data.filter(regex=r'(Amp|Torque)')
        denso_data['Time_Stamp'] = time['time']
        
        #If start and end time provided, slice the data accordingly
        if self.start != 'None':
            denso_data = denso_data[(denso_data['Time_Stamp'] > self.start) & (denso_data['Time_Stamp'] < self.end)]
        
        denso_data['Time_Stamp'] = pd.to_datetime(denso_data['Time_Stamp'])
        denso_data = denso_data.set_index('Time_Stamp')
        
        #amp_data = denso_data.filter(regex=r'(Amp)')
        #torq_data = denso_data.filter(regex=r'(Torq)')
        
        return denso_data
        
    def transforms(self, dataframe, agg=True, freq='1min', stat='mean', trans=True, trans_type='zscore'):
        """
        This is to transform/scale data. Choose zscore, minmax, or boxcox
        Defaults to using the parameters from class if a dataframe is not provided
        """       
        denso_data = dataframe
        
        if agg == True:
            #Resample for mean and max values
            if stat == 'max':
                denso_data = denso_data.resample(freq).max()
            elif stat == 'mean':
                denso_data = denso_data.resample(freq).mean()
                
        if trans == True:
                
            #Z-score Standardization
            if trans_type == 'zscore':
                denso_data = denso_data.apply(ss.zscore)

            
            #MinMax Standardization    
            elif trans_type =='minmax':
                trans = MinMaxScaler()
                array = trans.fit_transform(denso_data)
                denso_data = pd.DataFrame(array, columns=denso_data.columns, index=denso_data.index)
                
# =============================================================================
#             #BoxCox Standardization
#             elif trans_type =='boxcox':
#                 #Shift data since boxcox cannot transform negative data
#                 if denso_data.min().values[0] < 0:
#                     denso_data = denso_data + ((denso_data.min().values[0] * -1) + 0.0001)
#                 array, lam = ss.boxcox(denso_data[denso_data.columns].values)
#                 denso_data = pd.DataFrame(array, columns=denso_data.columns, index=denso_data.index)
#                 print('Lambda: %f' % lam)
# =============================================================================
                     
        return denso_data
        
    def kmeans(self, train_df, test_df):
        print('Finding Clusters')
        #Combine datasets and process together to same scale
        all_df = train_df.append(test_df)
        all_df = self.transforms(all_df, agg=False, trans=True, trans_type='zscore')
        
        #Loop through each joint to build 2-D dataframe (attribute for each joint)
        df_dict = {}
        for i in range(1,7):
            var = str(i)
            
            #Filter by Joint
            joint_df = all_df.filter(regex=var)
            
            #Create k-means unsupervised model (set to find 2 clusters)
            km = KMeans(n_clusters=2, init='random',n_init=10, max_iter=300, tol=1e-04, random_state=21, algorithm='lloyd')
            y_km = km.fit_predict(joint_df)
            
            #Create new column in dataframe with results and save DF to a dictionary to store all joints
            joint_df['labels'] = y_km
            df_dict[i] = joint_df
            df_reset = joint_df.reset_index()
            #Plot 2-D scatter plot of results
            plt.style.use('fivethirtyeight')
            plt.title(label= 'Joint' + str(i), fontsize=16, color="black")
            plt.xlabel(joint_df.columns[0])
            plt.ylabel(joint_df.columns[1])
            plt.scatter(x=joint_df.iloc[:,0], y=joint_df.iloc[:,1], c=joint_df.iloc[:,2], marker='.', cmap='Dark2')
            plt.show()
            plt.scatter(y=df_reset.iloc[:,2], x=df_reset.index.values, c=df_reset.iloc[:,3], marker='.', cmap='Dark2')
            plt.show()
            
        return df_reset
        
        
        
        
        
        