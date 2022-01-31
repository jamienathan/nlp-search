# APP 
from matplotlib.collections import LineCollection, PolyCollection
import streamlit as st

# Data Wranling
import numpy as np 
import pandas as pd
import re
from collections import Counter

# Data Visualisation
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.graphics.tsaplots import plot_acf
import plotly.io as pio
import wordcloud
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
from matplotlib_venn_wordcloud import venn2_wordcloud
from highlight_text import HighlightText, ax_text, fig_text

# NLP
from bertopic import BERTopic
import spacy
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob

# Load Spacy
nlp = spacy.load('en_core_web_trf')
nlp.max_length=2000000

# Page Setup
st.set_page_config(
    page_title='NLP Insights For Search', 
    initial_sidebar_state='expanded')

st.title('NLP Insights')


# UPLOAD FILE
st.sidebar.markdown('# Upload data!')
uploaded_file = st.sidebar.file_uploader(label='Upload csv here', type='csv')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df_questions = df.copy()

st.sidebar.markdown('# Setup')

# CHOOSE CORRECT COLUMNS
cols = df.columns.to_list()
searches = st.sidebar.selectbox('Which column are your queries in?', cols)
date = st.sidebar.selectbox('Which column are your dates in?', cols)
age = st.sidebar.selectbox('Which column is age in?', cols)
gender = st.sidebar.selectbox('Which column is gender in?', cols)

# ADD STOPWORDS
st.sidebar.markdown("""# Stopwords
Insert any stopwords below. These will be excluded from the analysis. Separate each word with a space. """)
custom_stopwords = st.sidebar.text_input(label='Insert any stopwords to exclude from analysis')
stopwords = stopwords.words('english')
stopwords.extend(custom_stopwords.split())

# CLEAN UPLOADED FILE
def clean_df(df: pd.DataFrame):
    """
    Returns
    --------
    Cleansed pd.DataFrame

    Function drops any queries which are null and replaces null demos with Unknown. 
    Removes punctuation, extra spaces, replaces underscores and words over 14 characters.
    Removes stopwords from the corpus. 
    """
    df = df.dropna(subset=[searches]).fillna('Unknown')
    df[date] = pd.to_datetime(df[date], format='%Y%m%d')
    df[searches] = (df[searches].str.replace('_', ' ')
                                .str.replace('(\S{14,})|[^\w\s]|  +', '') # Remove words > 14 letters, puncutation, numbers, doublespaces
                                .str.strip()
                                .str.lower()
                                .apply(lambda row: ' '.join([word for word in row.split() if word not in (stopwords)])))
    return df


df = clean_df(df)

st.subheader('Preview of upload')
st.write(df.head(10))

@st.cache(allow_output_mutation=True)
def get_topic_model(df):
    topic_model = BERTopic(
                    n_gram_range=(1,2), 
                    nr_topics="auto",
                    low_memory=True
                    )
    topics, topics = topic_model.fit_transform(df[searches].to_list())
    return topic_model, topics

@st.cache(allow_output_mutation=True)
def get_topic_map(topic_model):
    return topic_model.visualize_topics()

# topic_model, topics = get_topic_model(df)

# fig1 = get_topic_map(topic_model)
# st.write(fig1)


# SENTIMENT
st.header('Sentiment Analysis')
st.markdown(
    """
Searches with no sentiment expressed (i.e. **Neutral**) are removed from the analysis, leaving just the relative weight of positive and negative searches. 
""")

def sentiment_calc(text):
    return TextBlob(text).polarity

# Assign Sentiment
df['score'] = df[searches].apply(sentiment_calc)
df['sentiment'] = pd.cut(df['score'], bins=[-1, -0.2, 0.2, 1],labels=['Negative', 'Neutral', 'Positive'])

@st.cache(allow_output_mutation=True)
def sentiment_plot(df, offset='W-Mon'):
    # Typography and Colour 
    font = {'family' : 'Yahoo Sans', 'weight':'normal'}
    plt.rc('font', **font)
    sent_colors = {'Negative':'#E0E4E9', 'Neutral':'Grey', 'Positive':'#1AC567'}

    fig, ax = plt.subplots(figsize=(5,3))

    (df
    .set_index('date')                      
    .sort_index()                           
    .groupby(['sentiment'])['score']           
    .resample(offset)                       
    .sum()                                  
    .reset_index()
    .assign(pct=lambda df_:df_['score'] / df_['score'].sum() * 100)
    .pivot_table(index='date', columns='sentiment', values='pct', aggfunc='sum')
    .drop('Neutral', axis=1)
    .plot.area(color=sent_colors, legend=False, lw=0, ax=ax))
    
    # Axes
    ax.set(xlabel="")
    ax.set_ylabel('% of total sentiment', fontsize=6)
    ax.axhline(y=0, color='grey', zorder=10, lw=0.4)
    ax.tick_params(axis='both', which='both', bottom=False, left=False, labelsize=5)

    # Title
    s = 'Aggregate weight of <Positive> and <Negative> sentiment in searches'
    fig_text(0.05, 0.9, s, fontweight='bold', fontsize=10, va='bottom', highlight_textprops=[{"color": "#1AC567", "fontweight":'bold'},
                                                                        {"color": "#E0E4E9", "fontweight":"bold"}])

    # Aesthetics
    for direction in ['bottom', 'left']:
        ax.spines[direction].set_lw(0.2)
        ax.spines[direction].set_color('grey')
        ax.spines[direction].set_alpha(0.5)
    sns.despine(left=True, bottom=True)

    plt.savefig('sentiment.png', dpi=1000, transparent=True)

    return fig

# Date Aggregation Selection
def format_func(option):
    return CHOICES[option]

CHOICES = {'M':'Month', 'W-MON':'Week', 'Q':'Quarter', 'D':'Day', 'SM':'Bi-Week', 'B':'Business Day'}
offset = st.selectbox(label='Select Date Aggregation', options=list(CHOICES.keys()), format_func=format_func)

# Plot Sentiment
fig_sent = sentiment_plot(df, offset=offset)
st.pyplot(fig_sent, dpi=1000)

# Download Button
with open("sentiment.png", "rb") as file:
    btn = st.download_button(
        label="Download chart",
        data=file,
        file_name="sentiment.png",
        mime="image/png")


# ---------------

# QUESTIONS
st.header('Questions')
st.markdown("""
Any searches from your upload that contain questions are filtered here.  
It provides a sense of what people are researching or are unsure of.  
""")
# CSS to inject contained in a string
hide_dataframe_row_index = """
            <style>
            .row_heading.level0 {display:none}
            .blank {display:none}
            </style>
            """

# Inject CSS with Markdown
st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)

df_questions['length'] = df_questions[searches].str.len()
df_questions[searches] = df_questions[searches].str.replace('_', ' ').str.replace(r'(co.uk|www.|http://|\d| amp|&amp|(\S{14,}))', '').str.strip().str.replace('  ', ' ').str.lower()
st.dataframe(df_questions
        .dropna()
        .query('query.str.contains("^do |^how|what|when|why|where|^is ")', engine='python')
        .groupby('query', as_index=False)
        .size()
        .sort_values('size', ascending=False)
        .head(40)
        .filter(like='query'), width=800, height=600)

# ---------------
# WORDCLOUD

st.header('Parts of Speech')
st.markdown("""
Every word in the upload is analysed for it's part-of-speech, or role within the sentence. You can use the filters to pick out Nouns, Adjectives, Verbs or Pronouns, as well as layer in demographic profiles. 
By default all searches are included, but you can specify a demographic using the filters below. 
Any **stopwords** specified in the left hand navigation will apply to the WordCloud, so use this to remove unwanted words from the image. 
""")
# Colour format 
def purple(word=None, font_size=None,
                       position=None, orientation=None,
                       font_path=None, random_state=None):
    h = 267 # 0 - 360
    s = 99 # 0 - 100
    l = random_state.randint(50, 80) # 0 - 100
    return "hsl({}, {}%, {}%)".format(h, s, l)

def create_wordcloud(df):
    # Aesthetics
    font = {'family' : 'Yahoo Sans', 'weight' : 'bold'}
    plt.rc('font', **font)

    # Create spaCy document and tokenise 
    doc = nlp(" ".join(df.query('age in @age_selection and gender in @gender_selection')[searches].to_list()))
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct and token.pos_ in parts_option]
    tokens = Counter(tokens)

    # Instantiate wordcloud object
    wc = WordCloud(background_color='white', 
               color_func=purple,
               collocations=True,
               max_words=200,
               width=1200, height=1000, prefer_horizontal=0.9)

    wc = wc.generate_from_frequencies(tokens)
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.savefig('wordcloud.png', dpi=1000, transparent=True)
    return fig

# # Create copy of dataframe for use in wordcloud
# df_words = df.copy()

# # Demographic Selection
# ages = df[age].unique().tolist()
# genders = df[gender].unique().tolist()
# parts = ['ADJ', 'NOUN', 'VERB', 'PRON']

# # User Selectable Configuration
# with st.form(key='wordkey1'):
#     age_selection = st.multiselect('Select Age', ages, default=ages)
#     gender_selection = st.multiselect('Select Gender', genders, default=genders)
#     parts_option = st.multiselect(label='Select Part of Speech', options= parts, default=['NOUN'])
#     st.form_submit_button('Submit Choices') 

# # Plot Wordcloud
# fig_wordcloud = create_wordcloud(df_words)
# st.pyplot(fig_wordcloud, dpi=1000)

# # Download Button
# with open("wordcloud.png", "rb") as file:
#     btn = st.download_button(
#         label="Download wordcloud",
#         data=file,
#         file_name="wordcloud.png",
#         mime="image/png")

#-------------------

# // TODO WORDCLOUD VENN 
# st.subheader('WordCloud Venn')

# # User Selectable Configuration
# col1, col2 = st.columns(2)

# with st.form(key='vennkey'):
#     with col1:
#         age_venn1 = st.multiselect('Select Age', ages, default=ages, key='ven1age')
#         gender_venn1 = st.multiselect('Select Gender', genders, default=genders, key='ven1gen')
    
#     with col2:
#         age_venn2 = st.multiselect('Select Age', ages, default=ages, key='ven2age')
#         gender_venn2 = st.multiselect('Select Gender', genders, default=genders, key='ven2gen')
#     st.form_submit_button('Submit Choices') 

# doc1 = nlp(" ".join(df_words.query('age in @age_venn1 and gender in @gender_venn1')[searches].to_list()))
# tokens1 = [token.orth_ for token in doc1 if not token.is_stop and not token.is_punct and token.pos_ in parts_option]
# tokens1 = Counter(tokens1).most_common(25)

# doc2 = nlp(" ".join(df_words.query('age in @age_venn2 and gender in @gender_venn2')[searches].to_list()))
# tokens2 = [token.orth_ for token in doc2 if not token.is_stop and not token.is_punct and token.pos_ in parts_option]
# tokens2 = Counter(tokens2).most_common(25)

# # tokenize words (approximately at least):
# sets = []
# for strings in [tokens1, tokens2]:
#     sets.append(set(strings))

# word_to_frequency = tokens1 + tokens2

# fig_venn = venn2_wordcloud(sets, ax=ax)

# fig_venn

# fig, ax = plt.subplots(figsize=(8,8))
# st.pyplot(venn2_wordcloud(sets), ax=ax)


st.header('Topic Model') #// TODO Topic model
st.header('Dendrogram') #// TODO Dendrogram
st.header('Topics by Age') #// TODO Topic by age
st.header('Topics by Gender') #// TODO Topic by gender
st.header('Topics by Time') #// TODO Topic by Time
st.header('Aspect or Search Intent') # // TODO Aspect or search intent
st.header('Entity Recognition') #// TODO NER
st.header('Number of Words') # // TODO number of words
st.header('Day of Week Impact') #//TODO Topic Day impact
st.header('Time Series Decomposition') # // TODO Decomposition
st.header('Holiday and Custom Date Impact') #// TODO Holiday impact 

# AUTO CORRELATION
st.header('Autocorrelation') #// TODO Autocorrelation
st.markdown("""These plots graphically summarise the strength of a relationship between any given day and that of a day at prior time steps (lags).

An autocorrelation of **1** represents a perfect positive correlation, while **-1** represents a perfect negative correlation.
The purple shaded area is the 95% confidence interval - anything within here shows no significant correlation.""")

def autocorr_plot(ser, days=50):
    font = {'family' : 'Yahoo Sans', 'weight':'regular'}
    plt.rc('font', **font)

    fig, ax = plt.subplots(figsize=(5,3))
    daily_queries = df.groupby(date).size()
    plot_acf(x=daily_queries, 
             lags=days, 
             ax=ax, 
             use_vlines=True,
             missing='conservative',
             zero=False,
             auto_ylims=True,
             title="",
             vlines_kwargs={"colors": "#7E1FFF"})
    
    for item in ax.collections:
        if type(item)==PolyCollection:
            item.set_facecolor('#7E1FFF') #7E1FFF

        if type(item)==LineCollection:
            item.set_facecolor('#7E1FFF')
    
    for item in ax.lines:
        item.set_color('#7E1FFF')

 
    # Axes
    ax.set_xlabel("Days Lag", fontweight='bold', fontsize=8)
    ax.set_ylabel('Correlation', fontweight='bold', fontsize=8)
    ax.tick_params(axis='both', which='both', bottom=False, left=False, labelsize=8)
    # Title
    s = 'Autocorrelation of searches'
    fig_text(0.05, 0.95, s, fontweight='bold', fontsize=14, va='bottom', color='#6001D2')

    # Aesthetics
    for direction in ['bottom', 'left']:
        ax.spines[direction].set_lw(0.2)
        ax.spines[direction].set_color('grey')
        ax.spines[direction].set_alpha(0.5)
    sns.despine()
        
    
    plt.savefig('autocorr.png', dpi=1000, transparent=True)
    return fig

# Plot Autocorrelation
days = st.slider(label='Number of days', min_value=10, max_value=370, step=10, value=50, format='%d', help='mailto:jamie.nathan@yahooinc')
fig_autocorr = autocorr_plot(df[date], days=days)
st.pyplot(fig_autocorr, dpi=1000)

# Download Button
with open("autocorr.png", "rb") as file:
    btn = st.download_button(
        label="Download autocorrelation chart",
        data=file,
        file_name="autocorr.png",
        mime="image/png")



st.header('Emotion') #// TODO Emotion 


