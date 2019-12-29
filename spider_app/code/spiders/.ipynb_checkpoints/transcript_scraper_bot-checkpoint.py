import scrapy
from urllib.parse import urlparse
from slugify import slugify
import spider.spiderSettings as settings
from spider.pipelines import JsonWriterPipeline
from spider.pipelines import DataPipeline
import json
import pandas as pd
import random


