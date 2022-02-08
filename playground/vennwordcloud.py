# %%
import pandas as pd
# Interactive data app that builds a variety of visualisations from imported search data 
# Author: Jamie Nathan
# Date: 2022-02-01 

# APP 
from matplotlib.collections import LineCollection, PolyCollection
import streamlit as st

# Data Wrangling
import numpy as np 
import pandas as pd
import xlrd
import re
from collections import Counter

# Time Series
from prophet import Prophet
from statsmodels.graphics.tsaplots import plot_acf

# Data Visualisation
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotx
from highlight_text import HighlightText, ax_text, fig_text
import plotly.io as pio
import wordcloud
from matplotlib import font_manager
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from matplotlib_venn_wordcloud import venn2_wordcloud
from PIL import Image


# NLP
import spacy
import nltk
from bertopic import BERTopic
from nltk.corpus import stopwords
from textblob import TextBlob
# %%
venn = pd.read_csv('raw data/demo data.csv')
# %%
men = venn.query('gender == "Male"').loc[:, 'search']
women = venn.query('gender =="Female"').loc[:, 'search']

# %%
