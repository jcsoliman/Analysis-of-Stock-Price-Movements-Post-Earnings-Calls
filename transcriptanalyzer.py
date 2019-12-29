import os
import pandas as pd
import numpy as np
from pathlib import Path
import fnmatch as fm
import re
import textract
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime
from datafunctions import datafunctions as funcs
nltk.download('vader_lexicon')

class analyzetranscript:

    def __init__(self,resourcepath,fileextension='.pdf'):
        #super().__init__()
        self.resource_path=resourcepath
        self.file_extension=fileextension
        self.sentiments_data=[]
        self.tickers=[]
        self.sentiment_analyzer=SentimentIntensityAnalyzer()
        self.df_sentiment=pd.DataFrame(self.sentiments_data)

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

    def get_feature_values(self,col_start_index,col_end_index):
        return self.df_sentiment.iloc[:,col_start_index:col_end_index].values

    def get_target_values(self,target_column):
        return self.df_sentiment[target_column].values
    
    def __format_date__(self,strdate):
        return f'{strdate[0:2]}-{strdate[2:4]}-{strdate[4:]}'

    #Get distinct ticker names from downloaded transcripts
    def get_tickers(self):
        with os.scandir(self.resource_path) as files:
            for _file in files:
                if _file.is_file():
                    if fm.fnmatch(_file.name,'*'+self.file_extension):
                        #extract ticker name
                        result = re.match(r"[a-zA-z]+", _file.name.strip())
                        _filename=result.group(0).replace('_','')
                        #Check if the ticker is in the list. If not add to the list otherwise
                        #ignore the ticker
                        if self.check_if_exists(self.tickers,_filename):
                            self.tickers.append(_filename)

    #read transcript files and add transcript content to sentiment dataframe
    #https://realpython.com/working-with-files-in-python/
    def create_sentiment_data(self):
        encoding = 'utf-8'
        self.sentiments_data=[]
        for ticker in self.tickers:
            search_pattern=f'{ticker}*{self.file_extension}'
            with os.scandir(self.resource_path) as files:
                for _file in files:
                    if _file.is_file():
                        filename=_file.name.strip()
                        if fm.fnmatch(filename,search_pattern):
                            #Extract the content from PDF
                            #transcript_quarter=f'{(filename.split(".")[0]).split("_")[2]}:'
                            transcript_content=textract.process(f'{self.resource_path}/{filename}')
                            transcript_content=transcript_content.decode(encoding)
                            transcript_content=str(transcript_content).replace("\n", "").replace("\\", "")
                            #transcript_date=self.find_substring(transcript_content,transcript_quarter,9)
                            transcript_date=(filename.split(".")[0]).split("_")[3]

                            if isinstance(transcript_date,str):
                                
                                self.sentiments_data.append(
                                    self.get_sentiment_scores(
                                        transcript_content,
                                        self.__format_date__(transcript_date.strip()),
                                        ticker,
                                        filename
                                    )
                            )

        transcript_df=pd.DataFrame(self.sentiments_data)
        transcript_df.set_index(['Ticker','Date'],inplace=True)
        transcript_df=transcript_df.sort_values(by=['Ticker','Date'])
        #funcs.sort_values(transcript_df,['ticker','date'])
        self.df_sentiment=transcript_df
        return transcript_df

    def get_sentiment_scores(self,text, date, source, path):
        sentiment_scores = {}
        # Sentiment scoring with VADER
        text_sentiment = self.sentiment_analyzer.polarity_scores(text)
        sentiment_scores["Date"] = pd.to_datetime(date,format='%m-%d-%Y')
        #sentiment_scores["Text"] = text
        sentiment_scores["Ticker"] = source
        #sentiment_scores["Path"] = path
        #sentiment_scores["Compound"] = text_sentiment["compound"]
        sentiment_scores["Pos"] = text_sentiment["pos"]
        sentiment_scores["Neu"] = text_sentiment["neu"]
        sentiment_scores["Neg"] = text_sentiment["neg"]
        if text_sentiment["compound"] >= 0.05:  # Positive
            sentiment_scores["Normalized"] = 1
        elif text_sentiment["compound"] <= -0.05:  # Negative
            sentiment_scores["Normalized"] = -1
        else:
            sentiment_scores["Normalized"] = 0  # Neutral

        return sentiment_scores
    
