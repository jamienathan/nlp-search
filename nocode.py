# APP 
import streamlit as st

# Data Wranling
import numpy as np 
import pandas as pd
import re
from collections import Counter

# Data Visualisation
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.io as pio
import wordcloud
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
from matplotlib_venn_wordcloud import venn2_wordcloud

# NLP
from bertopic import BERTopic
import spacy
import nltk
from nltk.corpus import stopwords
from spacytextblob.spacytextblob import SpacyTextBlob

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


# QUESTIONS
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



# WORD CLOUD
st.subheader('WORDCLOUD')

# Demographic Selection
ages = df[age].unique().tolist()
genders = df[gender].unique().tolist()
parts = ['ADJ', 'NOUN', 'VERB', 'PRON']

# User Selectable Configuration
with st.form(key='wordkey'):
    age_selection = st.multiselect('Select Age', ages, default=ages)
    gender_selection = st.multiselect('Select Gender', genders, default=genders)
    parts_option = st.multiselect(label='Select Part of Speech', options= parts, default=['NOUN'])
    st.form_submit_button('Submit Choices') 



# Aesthetics
font = {'family' : 'Yahoo Sans',
        'weight' : 'bold'}
plt.rc('font', **font)

# Colour format 
def purple(word=None, font_size=None,
                       position=None, orientation=None,
                       font_path=None, random_state=None):
    h = 267 # 0 - 360
    s = 99 # 0 - 100
    l = random_state.randint(50, 80) # 0 - 100
    return "hsl({}, {}%, {}%)".format(h, s, l)


df_words = df.copy()

# Create spaCy document and tokenise 
doc = nlp(" ".join(df_words.query('age in @age_selection and gender in @gender_selection')[searches].to_list()))
tokens = [token.text for token in doc if not token.is_stop and not token.is_punct and token.pos_ in parts_option]
tokens = Counter(tokens)
clean_words = {key:value for key, value in Counter(tokens).items()}

# Instantiate wordcloud object
wc = WordCloud(background_color='white', 
               color_func=purple,
               collocations=True,
               max_words=200,
               width=1200, height=1000, prefer_horizontal=0.9
)

wc = wc.generate_from_frequencies(tokens)
fig, ax = plt.subplots(figsize=(8,8))
ax.imshow(wc, interpolation='bilinear')
plt.axis('off')
st.pyplot(fig)

# WORDCLOUD VENN
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

st.subheader('Sentiment Analysis')
st.header('TIME SERIES ANALYSIS')
st.subheader('Day of Week Impact')
st.subheader('Holiday and Custom Date Impact')
