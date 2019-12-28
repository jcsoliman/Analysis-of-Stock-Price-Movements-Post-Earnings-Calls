import pandas as pd

class datafunctions:
    def __init__(self):
        super().__init__()

    def concat_dataframes(self,dataframes,axisvalue='columns'):
        return pd.concat(dataframes,axis=axisvalue,join='inner').dropna()

    def calculate_standard_devation(self,dailyreturns,sortascending=True):
        return dailyreturns.std().sort_values(ascending=sortascending)

    def calculate_annulized_standard_devation(self,dailyreturns,workingdays=365,sortascending=True):
        return (dailyreturns * np.sqrt(workingdays)).sort_values(ascending=sortascending)
    
    def calculate_rolling_standard_deviation(self,dataframe,daywindow):
        return dataframe.rolling(window=daywindow).std()

    def calculate_correlation(self,dataframe):
        return dataframe.corr()
    
    def drop_columns(self,records,columns=[]):
        return records.drop(columns=columns,inplace=True)
    
    def rename_columns(self,records,columnmapper):
        return records.rename(columns=columnmapper,inplace=True)
    
    def sort_values(self,records,sort_columns=[],sortascending=True):
        return records.sort_values(by=sort_columns,ascending=sortascending,inplace=True)