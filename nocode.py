# APP 
from matplotlib.collections import LineCollection, PolyCollection
import streamlit as st

# Data Wranling
import numpy as np 
import pandas as pd
import re
from collections import Counter

# Time Series
from prophet import Prophet
from statsmodels.graphics.tsaplots import plot_acf


# Data Visualisation
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
@st.experimental_memo
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

# Get start and end dates for sourcing
research_start = df[date].dt.date.min().strftime('%-d %b %Y')
research_end = df[date].dt.date.max().strftime('%-d %b %Y')

st.subheader('Preview of upload')
st.write(df.head(10))

@st.experimental_memo
def get_topic_model(df):
    topic_model = BERTopic(
                    n_gram_range=(1,2), 
                    nr_topics="auto",
                    low_memory=True
                    )
    topics, topics = topic_model.fit_transform(df[searches].to_list())
    return topic_model, topics

@st.experimental_memo
def get_topic_map(topic_model):
    return topic_model.visualize_topics()

# topic_model, topics = get_topic_model(df)

# fig1 = get_topic_map(topic_model)
# st.write(fig1)

# ------------------------
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

@st.experimental_memo(persist='disk')
def sentiment_plot(df, offset='W-Mon'):
    # Typography and Colour 
    font = {'family' : 'Yahoo Sans', 'weight':'normal'}
    plt.rc('font', **font)
    sent_colors = {'Negative':'#E0E4E9', 'Neutral':'Grey', 'Positive':'#1AC567'}

    fig, ax = plt.subplots(figsize=(6,3))

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
    fig_text(0.05, 0.9, s, fontweight='bold', fontsize=12, va='bottom', highlight_textprops=[{"color": "#1AC567", "fontweight":'bold'},
                                                                        {"color": "#E0E4E9", "fontweight":"bold"}])

    # Caption
    fig.supxlabel(f'Source: Yahoo Internal (country) \nData covers {research_start} - {research_end}', fontsize=4, x=0.9, y=-0.05, ha='right')

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

@st.experimental_memo(persist='disk')
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

# Create copy of dataframe for use in wordcloud
df_words = df.copy()

# Demographic Selection
ages = df[age].unique().tolist()
genders = df[gender].unique().tolist()
parts = ['ADJ', 'NOUN', 'VERB', 'PRON']

# User Selectable Configuration
with st.form(key='wordkey1'):
    age_selection = st.multiselect('Select Age', ages, default=ages)
    gender_selection = st.multiselect('Select Gender', genders, default=genders)
    parts_option = st.multiselect(label='Select Part of Speech', options= parts, default=['NOUN'])
    st.form_submit_button('Submit Choices') 

# Plot Wordcloud
fig_wordcloud = create_wordcloud(df_words)
st.pyplot(fig_wordcloud, dpi=1000)

# Download Button
with open("wordcloud.png", "rb") as file:
    btn = st.download_button(
        label="Download wordcloud",
        data=file,
        file_name="wordcloud.png",
        mime="image/png")

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

# -----------------------------------------
# TIME SERIES DECOMPOSITION
st.header('Time Series Decomposition') # // TODO Decomposition
st.markdown("""These plots show the lift in searches for given days of the week, public holidays and custom dates. 
This is done by decomposing the provided data into three compontents (trend, seasonality & noise) to better understand patterns in the data. """)


def research_dates(df):
    return (df[date].dt.date.min().strftime('%-d %b %Y'), df[date].dt.date.max().strftime('%-d %b %Y'))


def is_spring_summer(ds):
    dt = pd.to_datetime(ds)
    return dt.quarter == 2 | dt.quarter == 3

holidays = pd.read_csv('raw data/custom_holidays.csv')
holidays['ds'] = pd.to_datetime(holidays['ds'], format='%d/%m/%Y')

def format_func_country(option):
    return COUNTRY_CHOICES[option]

st.subheader('Holiday and Custom Date Impact') #// TODO Holiday impact 
COUNTRY_CHOICES = {'UK':'UnitedKingdom', 'US': 'UnitedStates', 'AU':'Australia', 'AO': 'Angola', 'AR': 'Argentina', 'AW': 'Aruba', 'AT': 'Austria', 'AZ': 'Azerbaijan', 'BD': 'Bangladesh', 'BY': 'Belarus', 'BE': 'Belgium', 'BW': 'Botswana', 'BR': 'Brazil', 'BG': 'Bulgaria', 'BI': 'Burundi', 'CA': 'Canada', 'CL': 'Chile', 'CN': 'China', 'CO': 'Colombia', 'HR': 'Croatia', 'CW': 'Curacao', 'CZ': 'Czechia', 'DK': 'Denmark', 'DJ': 'Djibouti', 'DO': 'DominicanRepublic', 'EG': 'Egypt', 'EE': 'Estonia', 'ET': 'Ethiopia', 'ECB': 'EuropeanCentralBank', 'FI': 'Finland', 'FR': 'France', 'GE': 'Georgia', 'DE': 'Germany', 'GR': 'Greece', 'HN': 'Honduras', 'HK': 'HongKong', 'HU': 'Hungary', 'IS': 'Iceland', 'IN': 'India', 'IE': 'Ireland', 'IL': 'Israel', 'IT': 'Italy', 'JM': 'Jamaica', 'JP': 'Japan', 'KZ': 'Kazakhstan', 'KE': 'Kenya', 'KR': 'Korea', 'LV': 'Latvia', 'LS': 'Lesotho', 'LT': 'Lithuania', 'LU': 'Luxembourg', 'MY': 'Malaysia', 'MW': 'Malawi', 'MX': 'Mexico', 'MA': 'Morocco', 'MZ': 'Mozambique', 'NL': 'Netherlands', 'NA': 'Namibia', 'NZ': 'NewZealand', 'NI': 'Nicaragua', 'NG': 'Nigeria', 'MK': 'NorthMacedonia', 'NO': 'Norway', 'PY': 'Paraguay', 'PE': 'Peru', 'PL': 'Poland', 'PT': 'Portugal', 'PTE': 'PortugalExt', 'RO': 'Romania', 'RU': 'Russia', 'SA': 'SaudiArabia', 'RS': 'Serbia', 'SG': 'Singapore', 'SK': 'Slovakia', 'SI': 'Slovenia', 'ZA': 'SouthAfrica', 'ES': 'Spain', 'SZ': 'Swaziland', 'SE': 'Sweden', 'CH': 'Switzerland', 'TW': 'Taiwan', 'TR': 'Turkey', 'TN': 'Tunisia', 'UA': 'Ukraine', 'AE': 'UnitedArabEmirates', 'VE': 'Venezuela', 'VN': 'Vietnam', 'ZM': 'Zambia', 'ZW': 'Zimbabwe'}
country = st.selectbox(label='Select Holidays', options=list(COUNTRY_CHOICES.keys()), format_func=format_func_country)

with st.form("Enter Custom Date"):
    custom_name = st.text_input(label='Give the day you want to measure a name? i.e. Superbowl/CL Final')
    custom_date = st.date_input(label='What date was it on?')
    submitted = st.form_submit_button("Submit")



df_prophet = (df
                .groupby(date, as_index=False).size()
                .rename({date:'ds', 'size':'y'}, axis='columns')
                .sort_values('ds')
                .assign(autumn_winter=lambda df:~df['ds'].apply(is_spring_summer),
                        spring_summer=lambda df:df['ds'].apply(is_spring_summer)))

# Instantiate
m = Prophet(holidays=holidays, weekly_seasonality=False)

# Configure & Fit 
m.add_country_holidays(country_name=country) 
m.add_seasonality(name='weekly_springsummer', period=7, fourier_order=3, condition_name='spring_summer')
m.add_seasonality(name='weekly_autumnwinter', period=7, fourier_order=3, condition_name='autumn_winter')
m.fit(df_prophet)

# Predict
future = m.make_future_dataframe(periods=30)
future['autumn_winter'] = ~future['ds'].apply(is_spring_summer)
future['spring_summer'] = future['ds'].apply(is_spring_summer)

forecast = m.predict(future)
cols = ['ds', 'holidays', 'weekly_springsummer', 'weekly_autumnwinter', 'yhat']
df_seasonality = (pd.concat([forecast[cols], df_prophet['y']], axis='columns')
                    .dropna()
                    .assign(holiday_impact=lambda df: df['holidays'] / df['y'],
                                      ss_weeklyimpact=lambda df: df['weekly_springsummer'] / df['y'],
                                      aw_weeklyimpact=lambda df: df['weekly_autumnwinter'] / df['y'],
                                      dayofweek=lambda df: df['ds'].dt.day_name(),
                                      weekday=lambda df: df['ds'].dt.dayofweek))


def holiday_cleanup(ser):
    return (ser.str.replace('[England/Wales/Northern Ireland]', '', regex=False)
               .str.replace(' [Northern Ireland]', '', regex=False)
    )

df_holidays = (m.construct_holiday_dataframe(df_seasonality['ds'])
                .assign(holiday=lambda df: holiday_cleanup(df['holiday']))
                .set_index('ds'))

# holiday_impact = pd.concat([df_seasonality.set_index('ds'), df_holidays], axis='columns', join='inner').reset_index()

cols = ['ds', 'holidays', 'holiday_impact', 'yhat', 'y']

holiday_impact = (pd.concat([df_seasonality[cols].set_index('ds'), df_holidays], axis='columns')
                    .reset_index()
                    .query('holidays != 0')
                    .dropna(subset=['holidays'])
                    .fillna(''))

scot = holiday_impact['holiday'].str.contains("Scotland")

holiday_impact = holiday_impact.loc[~scot]


research_start, research_end = research_dates(df)

holiday_options = holiday_impact['holiday'].to_list()
holiday_selected = st.multiselect('Select & Deselect Holidays To Show In Chart', options=holiday_options, default=holiday_options)



@st.experimental_memo(persist='disk')
def holiday_impact_plot(df):
    fig, ax = plt.subplots(figsize=(6,3), dpi=1000)

    sns.set(font='Yahoo Sans', style='white')
    df['holiday_impact'] = df['holiday_impact'] * 100
    df.query('holiday in @holiday_selected').plot(x='holiday', y='holiday_impact', color=df['holiday_impact'].ge(0).map({True: '#7E1FFF', False:'lightgrey'}), kind='barh', width=1, ax=ax)

    # Labels
    ax.set_ylabel('')
    ax.set_xlabel('% lift in searches', fontsize=6, fontweight='bold')
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelsize=5, pad=0)

    # Spines
    sns.despine(left=True, bottom=False)

    for direction in ['bottom', 'left']:
        ax.spines[direction].set_lw(0.2)
        ax.spines[direction].set_color('grey')
        ax.spines[direction].set_alpha(0.5)
    
    ax.grid(which='major', axis='x', dashes=(1,3), zorder=4, color='gray', ls=':', alpha=0.5, lw=0.2)
    ax.grid(axis='y', visible=False)
    # ax.axvline(0, lw=0.2, color='black')
    # Title
    s = '% Lift in Searches by Holiday & Event'
    s2 = 'The chart shows the <Positive> and <Negative> impact on searches during holidays' 
    fig_text(0.00, 0.92, s, fontweight='bold', fontsize=12, va='bottom',  color='#6001D2')

    fig_text(0.00, 0.89, s2, fontsize=6, va='bottom', highlight_textprops=[{"color": "#7E1FFF", "fontweight":'bold'},
                                                                            {"color": "grey", "fontweight":"bold"}])

    # Adjust axes positions to fit commentary
    pos1 = ax.get_position() # get the original position 
    pos2 = [pos1.x0, pos1.y0 - 0.05,  pos1.width, pos1.height]
    ax.set_position(pos2)

    # Source
    fig.supxlabel(f'Source: Yahoo Internal {country} \nData covers {research_start} - {research_end}', fontsize=4, x=0.9, y=-0.05, ha='right')
    ax.get_legend().remove()
    plt.savefig('holiday_impact.png', dpi=1000, transparent=True)

    return fig

# Plot Holiday Impact
fig_holiday = holiday_impact_plot(holiday_impact)
st.pyplot(fig_holiday, dpi=1000)

# Download Button
with open("holiday_impact.png", "rb") as file:
    btn = st.download_button(
        label="Download holiday impact chart",
        data=file,
        file_name="holiday_impact.png",
        mime="image/png")

st.subheader('Day of Week Impact')
st.markdown("""Conditional seasonality is used to split **A/W** and **S/S** as behaviours tend to differ in each.  Taking the average over the course of a year hides these differences.""")

@st.experimental_memo(persist='disk')
def weekly_impact_plot(df):
        days = [ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        weekly_impact = (df_seasonality
                    .groupby('dayofweek')
                    .agg(ss=('weekly_springsummer', 'sum'),
                         aw=('weekly_autumnwinter', 'sum'),
                         pred=('yhat', 'sum'))
                    .reindex(days)
                    .assign(ss_impact=lambda df: df['ss'] / df['pred'] * 100,
                            aw_impact=lambda df: df['aw'] / df['pred'] * 100)
                    .reset_index())
                    
        seasonal_impact = (weekly_impact.melt(id_vars='dayofweek', var_name='season', value_vars=['ss_impact', 'aw_impact'], value_name='impact'))  

        sns.set(font='Yahoo Sans', style='white')
        fig, ax = plt.subplots(figsize=(6,3), dpi=1000)
        sns.barplot(x='dayofweek', y='impact', hue='season', palette=['#7E1FFF', 'lightgrey'], lw=0, ec='black', data=seasonal_impact, ax=ax);

        # Labels
        ax.xaxis.set_ticks_position('top')
        ax.set_xlabel('')
        ax.set_ylabel('% lift in searches', fontsize=6, fontweight='bold')
        ax.tick_params(axis='x', which='both', bottom=False, top=False, left=False, labelsize=5, labeltop=True, pad=-10)
        ax.tick_params(axis='y', which='both', bottom=False, top=False, left=False, labelsize=5, labeltop=True, pad=0)
        yticks = ax.get_yticks()

        # Spines
        sns.despine(left=True, bottom=True)

        for direction in ['bottom', 'left']:
                ax.spines[direction].set_lw(0.2)
                ax.spines[direction].set_color('grey')
                ax.spines[direction].set_alpha(0.5)

        ax.grid(which='major', axis='y', dashes=(1,3), zorder=4, color='gray', ls=':', alpha=0.5, lw=0.2)
        ax.grid(axis='x', visible=False)

        # Title
        s = '% Lift in Searches by Day of Week'
        s2 = 'The chart shows weekday impact during the <Spring/Summer> and <Autumn/Winter> after accounting for public holidays' 
        fig_text(0.05, 0.92, s, fontweight='bold', fontsize=12, va='bottom',  color='#6001D2')

        fig_text(0.05, 0.89, s2, fontsize=6, va='bottom', highlight_textprops=[{"color": "#7E1FFF", "fontweight":'bold'},
                                                                                {"color": "lightgrey", "fontweight":"bold"}])

        
        # Caption
        fig.supxlabel(f'Source: Yahoo Internal {country} \nData covers {research_start} - {research_end}', fontsize=4, x=0.9, ha='right')

        # Custom Grouping Aesthetics
        for day in range(7):
                ax.plot([day - 0.45, day + 0.45], [yticks[-1], yticks[-1]], color='black', lw=0.5)

        ax.get_legend().remove()

        # Adjust axes positions to fit commentary
        pos1 = ax.get_position() # get the original position 
        pos2 = [pos1.x0, pos1.y0 - 0.05,  pos1.width, pos1.height]
        ax.set_position(pos2)

        plt.savefig('weekly_impact.png', dpi=1000, transparent=True)

        return fig

# Plot Daily Impact
fig_weekly = weekly_impact_plot(df_seasonality)
st.pyplot(fig_weekly, dpi=1000)

# Download Button
with open("weekly_impact.png", "rb") as file:
    btn = st.download_button(
        label="Download weekly impact chart",
        data=file,
        file_name="weekly_impact.png",
        mime="image/png")


# -----------------------------------------
# AUTO CORRELATION
st.header('Autocorrelation') #// TODO Autocorrelation
st.markdown("""These plots graphically summarise the strength of a relationship between any given day and that of a day at prior time steps (lags).

An autocorrelation of **1** represents a perfect positive correlation, while **-1** represents a perfect negative correlation.
The purple shaded area is the 95% confidence interval - anything within here shows no significant correlation.""")

@st.experimental_memo(persist='disk')
def autocorr_plot(ser, days=50):
    font = {'family' : 'Yahoo Sans', 'weight':'regular'}
    plt.rc('font', **font)

    fig, ax = plt.subplots(figsize=(6,3))
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
    ax.tick_params(axis='both', which='both', bottom=False, left=False, labelsize=8, pad=0)
    # Title
    s = 'Autocorrelation of searches'
    fig_text(0.05, 0.95, s, fontweight='bold', fontsize=14, va='bottom', color='#6001D2')

    # Caption
    fig.supxlabel(f'Source: Yahoo Internal {country} \nData covers {research_start} - {research_end}', fontsize=4, x=0.9, y=-0.05, ha='right')

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


