import os
import pandas as pd
import numpy as np
from pathlib import Path
import fnmatch as fm
import re
import textract
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

class analyzetranscript:

    def __init__(self,resourcepath,fileextension='.pdf'):
        #super().__init__()
        self.resource_path=resourcepath
        self.file_extension=fileextension
        self.sentiments_data=[]
        self.tickers=[]
        self.sentiment_analyzer=SentimentIntensityAnalyzer()
    
    def concat_dataframes(self,dataframes,axisvalue):
        return pd.concat(dataframes,axis=axisvalue,join='inner').dropna()

    def calculate_standard_devation(self,dailyreturns,sortascending=True):
        return dailyreturns.std().sort_values(ascending=sortascending)

    def calculate_annulized_standard_devation(self,dailyreturns,workingdays=365,sortascending=True):
        return (dailyreturns * np.sqrt(workingdays)).sort_values(ascending=sortascending)
    
    def calculate_rolling_standard_deviation(self,dataframe,daywindow):
        return dataframe.rolling(window=daywindow).std()

    def calculate_correlation(self,dataframe):
        return dataframe.corr()

    def find_substring(self, strText, strSubString, Offset=None):
        try:
            Start = strText.find(strSubString)
            if Start == -1:
                return -1 # Not Found
            else:
                if Offset == None:
                    Result = strText[Start+len(strSubString):]
                elif Offset == 0:
                    return Start
                else:
                    AfterSubString = Start+len(strSubString)
                    Result = strText[AfterSubString:AfterSubString + int(Offset)]
                return Result
        except:
            return -1

    def check_if_exists(self,items,item):
        return item not in items

    #Get distinct ticker names from downloaded transcripts
    def get_tickers(self):
        for file_name in os.listdir(self.resource_path):
            if fm.fnmatch(file_name,'*'+self.file_extension):
                #extract ticker name
                result = re.match(r"[a-zA-z]+", file_name)
                _filename=result.group(0).replace('_','')
                #Check if the ticker is in the list. If not add to the list otherwise
                #ignore the ticker
                if self.check_if_exists(self.tickers,_filename):
                    self.tickers.append(_filename)

    #read transcript files and add transcript content to sentiment dataframe
    #https://realpython.com/working-with-files-in-python/
    def create_sentiment_data(self):
        encoding = 'utf-8'
        
        for ticker in self.tickers:
            search_pattern=f'{ticker}*{self.file_extension}'
            for filename in os.listdir(self.resource_path):
                if fm.fnmatch(filename,search_pattern):
                    #Extract the content from PDF
                    transcript_content=textract.process(f'{self.resource_path}/{filename}')
                    transcript_content=transcript_content.decode(encoding)
                    transcript_content=str(transcript_content).replace("\n", "").replace("\\", "")
                    transcript_date=''
                    self.sentiments_data.append(
                        self.get_sentiment_scores(
                            transcript_content,
                            transcript_date,
                            ticker,
                            filename
                        )
                    )

        transcript_df=pd.DataFrame(self.sentiments_data)
        transcript_df=transcript_df.sort_values(by=['source','date'])
        transcript_df.set_index('date',inplace=True)
        return transcript_df

    def get_sentiment_scores(self,text, date, source, path):
        sentiment_scores = {}
        # Sentiment scoring with VADER
        text_sentiment = self.sentiment_analyzer.polarity_scores(text)
        sentiment_scores["date"] = date
        sentiment_scores["text"] = text
        sentiment_scores["source"] = source
        sentiment_scores["path"] = path
        sentiment_scores["compound"] = text_sentiment["compound"]
        sentiment_scores["pos"] = text_sentiment["pos"]
        sentiment_scores["neu"] = text_sentiment["neu"]
        sentiment_scores["neg"] = text_sentiment["neg"]
        if text_sentiment["compound"] >= 0.05:  # Positive
            sentiment_scores["normalized"] = 1
        elif text_sentiment["compound"] <= -0.05:  # Negative
            sentiment_scores["normalized"] = -1
        else:
            sentiment_scores["normalized"] = 0  # Neutral

        return sentiment_scores
    
