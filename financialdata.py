import os
import pandas as pd
import simfin as sf
from simfin.names import *
import requests as r
from datafunctions import datafunctions as funcs

class financial_data:
    def __init__(self,tickers,sentiment_data,start_year=2013,end_year=2019):
        super().__init__()
        self.tickers=tickers
        self.sentiment_data=sentiment_data
        self.api_key=os.getenv('SIMFIN_API')
        self.stock_ids=[]
        self.time_periods=['Q1','Q2','Q3','Q4']
        self.year_start=start_year
        self.year_end=end_year
        self.columns_to_drop=[]
        self.sort_on_index=[]
        self.sort_on_columns=[]
        self.df_prices=pd.DataFrame()
    
    def __get_ids__(self):
        #get id for the list of tickers
        for ticker in self.tickers:
            request_url=f'https://simfin.com/api/v1/info/find-id/ticker/{ticker}?api-key={api_key}'
            content = r.get(request_url)
            data = content.json()

            if "error" in data or len(data) < 1:
                stock_ids.append(None)
            else:
                stock_ids.append(data[0]['simId'])

    def __load_share_data__(self):
        sf.set_data_dir('~/simfin_data/')
        self.df_prices = sf.load_shareprices(variant='daily', market='US')
    
    def get_share_prices(self):
        if self.df_prices.empty == True:
            self.__load_share_data__()
        #funcs.drop_columns(df_prices,self.columns_to_drop)
        self.df_prices.reset_index(inplace=True)
        self.df_prices.drop(columns=self.columns_to_drop,inplace=True)
        self.df_prices['Returns'] = self.df_prices['Close'].pct_change()
        self.df_prices['12-day Rolling']= self.df_prices['Returns'].rolling(window=12).mean()
        self.df_prices['5-day Rolling']= self.df_prices['Returns'].rolling(window=5).mean()
        self.df_prices['3-day Rolling']= self.df_prices['Returns'].rolling(window=3).mean()
        self.df_prices['12-day Rolling Std']= self.df_prices['Returns'].rolling(window=12).std()
        self.df_prices['5-day Rolling Std']= self.df_prices['Returns'].rolling(window=5).std()
        self.df_prices['3-day Rolling Std']= self.df_prices['Returns'].rolling(window=3).std()
        self.df_prices= self.df_prices[self.df_prices.Ticker.str.contains('|'.join(self.tickers))].set_index(['Ticker','Date'])
        self.df_prices.sort_values(by=self.sort_on_index,inplace=True)
        #funcs.sort_values(self.df_prices,self.sort_on_index)
        df= pd.concat([self.sentiment_data,self.df_prices],axis=1,join='inner').dropna()
        return df.sort_index()
    

