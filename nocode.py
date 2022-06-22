# Interactive data app that builds a variety of visualisations from imported search data 
# Author: Jamie Nathan
# Date: 2022-02-01 

# APP 
from concurrent.futures import process
from matplotlib.collections import LineCollection, PolyCollection
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

# Data Wrangling
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
import matplotx
from highlight_text import HighlightText, ax_text, fig_text
import plotly.io as pio
import wordcloud
from matplotlib import font_manager, gridspec
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from matplotlib_venn_wordcloud import venn2_wordcloud
from PIL import Image
from scipy.cluster.hierarchy import linkage
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.figure_factory as ff
from itertools import cycle


# NLP
import spacy
import nltk
nltk.download('stopwords')
from bertopic import BERTopic
from nltk.corpus import stopwords
from textblob import TextBlob
from spacy_streamlit import process_text, load_model


# PAGE SETUP & AESTHETICS

st.set_page_config(
    page_title='NLP Insights For Search', 
    initial_sidebar_state='expanded',
    layout='wide')

st.title('NLP INSIGHTS FOR SEARCH')

font_dirs = ['fonts/Yahoo Sans']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
    
plt.rcParams['font.family'] = 'Yahoo Sans'


# FUNCTIONS

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
    df = (df.rename({search:'search', age:'age', gender:'gender', date:'date'}, axis=1)
            .loc[:, ['date', 'search', 'age', 'gender']]
            .dropna(subset=['search'])
            .fillna('Unknown')
            .assign(gender=lambda df: df['gender'].str.lower(),
                    date=pd.to_datetime(df['date'], format='%Y%m%d'),
                    search=lambda df: df['search'].str.replace('_', ' ')
                                                  .str.replace('(\S{14,})|[^\w\s]|  +', '') 
                                                  .str.strip()
                                                  .str.lower()))
    return df


# DEMO
def demo_barplot(df):

        yahoo_palette_strict = "#39007D #7E1FFF #907CFF #003ABC #0F69FF #7DCBFF #C7CDD2 #828A93".split()

        age_barplot = (df.query('age in @age_selection')
                        .groupby('age', as_index=False)
                        .agg(total=('search', 'size'))
                        .assign(perc=lambda df: (df['total'] / df['total'].sum() * 100).round(1))
                        )

        gender_barplot = (df.query('gender in @gender_selection')
                        .groupby('gender', as_index=False)
                        .agg(total=('search', 'size'))
                        .assign(perc=lambda df: (df['total'] / df['total'].sum() * 100).round(1))
                        )
                        
        fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(6,2),dpi=500, gridspec_kw={'width_ratios': [1, 3]})    

        sns.barplot(x = 'gender', y = 'perc',data=gender_barplot, palette=['#7E1FFF', '#C7CDD2'], ax=ax[0])
        sns.barplot(x = 'age', y = 'perc',data=age_barplot, palette=yahoo_palette_strict, ax=ax[1])

        # Axes
        for plot in [0, 1]:
            ax[plot].set_ylabel('')
            ax[plot].set_xlabel('')
            ax[plot].set_yticklabels('')
            ax[plot].bar_label(ax[plot].containers[0], size=7, color='white', weight='bold', label_type='edge', padding=-20, fmt='%.0f')
            ax[plot].tick_params(axis='both', which='both', bottom=False, top=False, right=False, left=False, labelsize=7, pad=0),

        sns.despine(bottom=True, left=True)

        # Title
        fig.text(x=0.12, y=0.95, s='Demographic profile of searchers', fontweight='extra bold', fontsize=16, va='bottom',  color='#6001D2')

        # Sourcing
        fig.supxlabel(f'Source: Yahoo Internal - {market} \nData covers {research_start} - {research_end}', fontsize=4, x=0.9,y=-0.2, ha='right')

        plt.savefig('images/demo_barplot.png', transparent=True, dpi=1000, bbox_inches='tight')

        return fig


# TOPIC MODELLING
#@st.experimental_memo(persist='disk')
def get_topic_model(df):
    df['search'] = df['search'].apply(lambda row: ' '.join([word for word in row.split() if word not in (stopwords)]))
    topic_model = BERTopic(
                    language=user_language,
                    n_gram_range=(1,2), 
                    nr_topics=nr_topics,
                    low_memory=True,
                    calculate_probabilities=False
                    )
    topics, probs = topic_model.fit_transform(df['search'].to_list())
    return topic_model, topics

def visualize_topics(topic_model,
                     topics: List[int] = None,
                     top_n_topics: int = None,
                     width: int = 1300,
                     height: int = 700) -> go.Figure:
    """ Visualize topics, their sizes, and their corresponding words

    This visualization is highly inspired by LDAvis, a great visualization
    technique typically reserved for LDA.

    Arguments:
        topic_model: A fitted BERTopic instance.
        topics: A selection of topics to visualize
        top_n_topics: Only select the top n most frequent topics
        width: The width of the figure.
        height: The height of the figure.

    Usage:

    To visualize the topics simply run:

    ```python
    topic_model.visualize_topics()
    ```

    Or if you want to save the resulting figure:

    ```python
    fig = topic_model.visualize_topics()
    fig.write_html("path/to/file.html")
    ```
    <iframe src="../../getting_started/visualization/viz.html"
    style="width:1000px; height: 680px; border: 0px;""></iframe>
    """
    # Select topics based on top_n and topics args
    if topics is not None:
        topics = list(topics)
    elif top_n_topics is not None:
        topics = sorted(topic_model.get_topic_freq().Topic.to_list()[1:top_n_topics + 1])
    else:
        topics = sorted(list(topic_model.get_topics().keys()))

    # Extract topic words and their frequencies
    topic_list = sorted(topics)
    frequencies = [topic_model.topic_sizes[topic] for topic in topic_list]
    words = [" | ".join([word[0] for word in topic_model.get_topic(topic)[:5]]) for topic in topic_list]

    # Embed c-TF-IDF into 2D
    all_topics = sorted(list(topic_model.get_topics().keys()))
    indices = np.array([all_topics.index(topic) for topic in topics])
    embeddings = topic_model.c_tf_idf.toarray()[indices]
    embeddings = MinMaxScaler().fit_transform(embeddings)
    embeddings = UMAP(n_neighbors=2, n_components=2, metric='hellinger').fit_transform(embeddings)

    # Visualize with plotly
    df = pd.DataFrame({"x": embeddings[1:, 0], "y": embeddings[1:, 1],
                       "Topic": topic_list[1:], "Words": words[1:], "Size": frequencies[1:]})
    return _plotly_topic_visualization(df, topic_list, width, height)


def _plotly_topic_visualization(df: pd.DataFrame,
                                topic_list: List[str],
                                width: int,
                                height: int):
    """ Create plotly-based visualization of topics with a slider for topic selection """

    def get_color(topic_selected):
        if topic_selected == -1:
            marker_color = ["white" for _ in topic_list[1:]]
        else:
            marker_color = ["#6001D2" if topic == topic_selected else "white" for topic in topic_list[1:]]
        return [{'marker.color': [marker_color]}]

    # Prepare figure range
    x_range = (df.x.min() - abs((df.x.min()) * .15), df.x.max() + abs((df.x.max()) * .15))
    y_range = (df.y.min() - abs((df.y.min()) * .15), df.y.max() + abs((df.y.max()) * .15))

    # Plot topics
    fig = px.scatter(df, x="x", y="y", size="Size", size_max=40, template="simple_white", labels={"x": "", "y": ""},
                     hover_data={"Topic": True, "Words": True, "Size": True, "x": False, "y": False})
    fig.update_traces(marker=dict(color="white", line=dict(width=2.5, color='#6001D2')))

    # Update hover order
    fig.update_traces(hovertemplate="<br>".join(["<b>Topic %{customdata[0]}</b>",
                                                 "Words: %{customdata[1]}",
                                                 "Size: %{customdata[2]}"]))

    # Create a slider for topic selection
    steps = [dict(label=f"Topic {topic}", method="update", args=get_color(topic)) for topic in topic_list[1:]]
    sliders = [dict(active=0, pad={"t": 50}, steps=steps)]

    # Stylize layout
    fig.update_layout(
        title={
            'text': "<b>Topic Clustering of Search Queries",
            'y': .98,
            'x': 0.4,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=50,
                color="#6001D2",
		family="Yahoo Sans")
        },
        width=width,
        height=height,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Yahoo Sans"
        ),
        xaxis={"visible": False},
        yaxis={"visible": False},
        sliders=sliders
    )

    # Update axes ranges
    fig.update_xaxes(range=x_range)
    fig.update_yaxes(range=y_range)

    # Add grid in a 'plus' shape
    fig.add_shape(type="line",
                  x0=sum(x_range) / 2, y0=y_range[0], x1=sum(x_range) / 2, y1=y_range[1],
                  line=dict(color="white", width=2)) #CFD8DC
    fig.add_shape(type="line",
                  x0=x_range[0], y0=sum(y_range) / 2, x1=x_range[1], y1=sum(y_range) / 2,
                  line=dict(color="white", width=2)) #9E9E9E
    # fig.add_annotation(x=x_range[0], y=sum(y_range) / 2, text="D1", showarrow=False, yshift=10)
    # fig.add_annotation(y=y_range[1], x=sum(x_range) / 2, text="D2", showarrow=False, xshift=10)
    fig.data = fig.data[::-1]

    return fig


# @st.experimental_memo(persist='disk')
def get_topic_map(_topic_model):
    fig = topic_model.visualize_topics(width=1500, height=700) 
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    })

    return fig

def topic_bars(topic_model):
    fig = topic_model.visualize_barchart(width=350, top_n_topics=16)
    fig.update_xaxes(showgrid=False, tickfont=dict(size=10, color='black', family='Yahoo Sans Light'))
    fig.update_yaxes(showgrid=False, tickfont=dict(size=14, color='black', family='Yahoo Sans Light'))
    fig.update_layout(
        plot_bgcolor='white',
        grid_columns=3,

        title={
            'text': '<b>Topic Descriptions',
            'y': 0.98,
            'x': 0.2,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=50,
                color="#6001D2",
              family="Yahoo Sans")
        })
    
    palette = cycle(['#39007D', '#6001D2', '#907CFF', '#003ABC', '#0F69FF', '#12A9FF', '#00873C', '#1AC567', '#21D87D', '#FFD333', '#FF0080', '#C7CDD2', '#828A93', '#2C363F', '#39007D', '#6001D2'])

    for bar in fig.data:
        bar.marker.color = next(palette)
        bar.marker.opacity = 1

    for title in fig['layout']['annotations']:
        title['font'] = dict(family='Yahoo Sans', size=20)
        title.font.color = next(palette)

    return fig

def visualize_hierarchy(topic_model,
                            orientation: str = "left",
                            topics: List[int] = None,
                            top_n_topics: int = None,
                            width: int = 1300,
                            height: int = 700) -> go.Figure:
        

        # Select topic embeddings
        if topic_model.topic_embeddings is not None:
            embeddings = np.array(topic_model.topic_embeddings)
        else:
            embeddings = topic_model.c_tf_idf

        # Select topics based on top_n and topics args
        if topics is not None:
            topics = sorted(list(topics))
        elif top_n_topics is not None:
            topics = sorted(topic_model.get_topic_freq().Topic.to_list()[1:top_n_topics + 1])
        else:
            topics = sorted(list(topic_model.get_topics().keys()))

        # Select embeddings
        all_topics = sorted(list(topic_model.get_topics().keys()))
        indices = np.array([all_topics.index(topic) for topic in topics])
        embeddings = embeddings[indices]

        # Create dendogram
        distance_matrix = 1 - cosine_similarity(embeddings)
        fig = ff.create_dendrogram(distance_matrix,
                                orientation=orientation,
                                linkagefun=lambda x: linkage(x, "ward"),
                                color_threshold=1,
                                colorscale=yahoo_palette_strict)

        # Create nicer labels
        axis = "yaxis" if orientation == "left" else "xaxis"
        new_labels = [[[str(topics[int(x)]), None]] + topic_model.get_topic(topics[int(x)])
                    for x in fig.layout[axis]["ticktext"]]
        new_labels = ["_".join([label[0] for label in labels[1:3]]) for labels in new_labels]
        new_labels = [label if len(label) < 30 else label[:27] + "..." for label in new_labels]
        custom_labels = topic_model.get_topic_info().Name.str.split("_", ).str.get(1)

        # Stylize layout
        fig.update_layout(
            plot_bgcolor='white',
            template="plotly_white",
            title={
                'text': "<b>Hierarchical Clustering of Search Queries",
                'x': 0,
                'xanchor': 'left',
                'yanchor': 'top',
                'font': dict(
                    size=60,
                    color="#6001D2",
                    family='Yahoo Sans')
            },
            hoverlabel=dict(
                bgcolor="white",    
                font_size=16,
                font_family="Yahoo Sans"
            ),
        )
        fig.update_yaxes(visible=False)
        fig.update_xaxes(ticks="", showline=False,tickfont=dict(size=13, 
                                                color='black', 
                                                family='Yahoo Sans'))


        # Stylize orientation
        if orientation == "left":
            fig.update_layout(height=200+(15*len(topics)),
                            width=width,
                            yaxis=dict(tickmode="array",
                                        ticktext=new_labels))
            
            # Fix empty space on the bottom of the graph
            y_max = max([trace['y'].max()+5 for trace in fig['data']])
            y_min = min([trace['y'].min()-5 for trace in fig['data']])
            fig.update_layout(yaxis=dict(range=[y_min, y_max]))

        else:
            fig.update_layout(width=700+(15*len(topics)),
                            height=height,
                            xaxis=dict(tickmode="array",
                                        ticktext=new_labels,
                                        tickangle=-60))
        return fig



# @st.experimental_memo
def topic_facet(searches, topics, timestamps):
    topic_list = topic_model.get_topic_info()

    # Create Dynamic Topic Model and Merge topic names for readability
    topics_over_time = (topic_model
                    .topics_over_time(df['search'], topics, timestamps=df['date'])
                    .query('Topic >= 0 and Topic < 16')
                    .sort_values(['Topic', 'Frequency'])
                    .merge(topic_list[['Topic', 'Name']], how='left', on='Topic')
                    .assign(Topic=lambda df: df['Topic'].astype(str).str.zfill(2))
                    .assign(Topic=lambda df: df['Topic'] + " " + df['Name'].str.split('_').str[1:3].str.join(' ')))

    # Resample the data at the monthly level to provide smoother view. Filter just the top 16 topics for performance. 
    times = (topics_over_time
            .query('"16" > Topic >= "00"')
            .groupby(['Name', 'Topic', pd.Grouper(freq='M', key='Timestamp')])
            .agg(cnt=('Frequency', 'sum'))
            .fillna(0)
            .reset_index())

    # Sort Topics by count to pass to Facetgrid sort
    sort_order = times.groupby('Topic', as_index=False).sum('cnt').Topic.to_list()
    sns.set(font='Yahoo Sans', context='paper', style='white', rc={'figure.figsize':(7,2), 'font.weight':'normal'})

    g = sns.relplot(
        data=times,
        x="Timestamp", y="cnt", col="Topic",
        kind="line", color='#6001D2', linewidth=4, zorder=5, col_order=sort_order,
        col_wrap=4, height=2, aspect=1.5, legend=False, facet_kws={'sharex':True},
    )

    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)

    for ax in g.axes.flatten():
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontweight('bold')

    # Iterate over each subplot to customize further
    for topic, ax in g.axes_dict.items():
        sns.set(font='Yahoo Sans', context='talk', style='white', rc={'font.weight':'normal'})

        # Add the title as an annotation within the plot
        heading = topic[:30] + '..' if len(topic) > 30 else topic
        ax.text(0, 1, heading.title(), transform=ax.transAxes, fontsize=8, fontweight="bold", fontfamily='Yahoo Sans', color='#6001D2')

        # Plot every year's time series in the background
        sns.lineplot(
            data=times, x="Timestamp", y="cnt", units='Topic', 
            estimator=None, color=".7", linewidth=1, alpha=0.2, ax=ax,
        )

        sns.despine(left=True)

        for direction in ['bottom']:
            ax.spines[direction].set_lw(0.2)
            ax.spines[direction].set_color('grey')
            ax.spines[direction].set_alpha(0.5)

    # Title
    g.fig.text(x=0, y=0.93, s='Topic Volumes Over Time', fontweight='extra bold', fontsize=35, va='bottom',  color='#6001D2')
    fig_text(x=0, y=0.90, s='Each <topic> is plotted against the relative volumes of <other topics> for comparison', fontsize=14, fontweight='normal', va='bottom', highlight_textprops=[{"color": "#6001D2", "fontweight":'bold'},
                                                                            {"color": "grey", "fontweight":"bold"}])
        
    # Sourcing
    g.fig.supxlabel(f'Source: Yahoo Internal - {market}\nData covers {research_start} - {research_end}', fontsize=8, x=1,y=0, ha='right')

    # Tweak the supporting aspects of the plot
    ax.set_yticks([])
    g.set_titles("")
    g.set_axis_labels("", "Freq.", )
    g.tight_layout(w_pad=-0)
    g.fig.subplots_adjust(top=0.85, left=0.05)

    plt.savefig('images/topic_time.png', transparent=True, dpi=1000)
    
    return g

def topics_demo(df, topics):
  # gender_classes = df['gender'].to_list()
  age_classes = df.query('age not in ["unknown", "Unknown", "Below13", "13-17"]')['age'].unique().tolist()
  topics_by_age = topic_model.topics_per_class(df['search'], topics, df['age'].to_list())
  # topics_by_gender = topic_model.topics_per_class(df['search'], topics, gender_classes)
  # topics_by_class = pd.concat([topics_by_age, topics_by_gender])

  left_bars = (topics_by_age
              .loc[topics_by_age['Class'] == "35-54"]
              .query('Topic != -1')
              .groupby(['Topic', 'Words'], as_index=False)
              .agg({'Frequency':'sum'})
              .sort_values('Frequency', ascending=False)
              .head(15)
              .sort_values('Frequency', ascending=False)
              .assign(new_topic=(lambda df:df['Words'].str.split(',').str[0:2].str.join(',').astype('category')))
  )

  right_bars = (topics_by_age
                .loc[topics_by_age['Class'] == "55+"]
                .query('Topic != -1')
                .groupby(['Topic', 'Words'], as_index=False)
                .agg({'Frequency':'sum'})
                .sort_values('Frequency', ascending=False)
                .head(15)
                .sort_values('Frequency', ascending=False)
                .assign(new_topic=(lambda df:df['Words'].str.split(',').str[0:2].str.join(',').astype('category')))
                )


  fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6,3), dpi=600, tight_layout=True)
  sns.set(font='Yahoo Sans', style='white', font_scale=0.5)
  sns.barplot(x='Frequency', y='new_topic', data=left_bars, ax=ax[0], color=['#7E1FFF'], orient='h', dodge=False, order=left_bars['new_topic'])
  sns.barplot(x='Frequency', y='new_topic', data=right_bars, ax=ax[1], color=['#C7CDD2'], orient='h', dodge=False, order=right_bars['new_topic'])

  for plot in [0, 1]:
    ax[plot].set(xlabel="", ylabel="")
    ax[plot].tick_params(axis='both', which='both', bottom=False, top=False, right=False, left=False, labelsize=7, pad=-3)
    ax[plot].set_yticklabels(ax[plot].get_yticklabels(), size = 5)
    ax[plot].set_xticklabels("")

  # Title
  fig.text(x=0.12, y=0.95, s='Top Topics by Demographic', fontweight='extra bold', fontsize=16, va='bottom',  color='#6001D2')

  # Sourcing
  fig.supxlabel(f'Source: Yahoo Internal - {market} \nData covers {research_start} - {research_end}', fontsize=4, x=0.9,y=-0.2, ha='right')

  plt.savefig('images/topic_barplot.png', transparent=True, dpi=1000, bbox_inches='tight')

  sns.despine(left=True, bottom=True)

  fig.subplots_adjust(top=0.80)

  return fig


# SENTIMENT

def sentiment_calc(text):
        try:
            return TextBlob(text).polarity
        except:
            return None

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
                                                                            {"color": "#E0E4E9", "fontweight":"extra bold"}])

        # Caption
        fig.supxlabel(f'Source: Yahoo Internal - {market} \nData covers {research_start} - {research_end}', fontsize=4, x=0.9, y=-0.05, ha='right')

        # Aesthetics
        for direction in ['bottom', 'left']:
            ax.spines[direction].set_lw(0.2)
            ax.spines[direction].set_color('grey')
            ax.spines[direction].set_alpha(0.5)
        sns.despine(left=True, bottom=True)

        plt.savefig('images/sentiment.png', dpi=1000, transparent=True)

        return fig


# WORDCLOUD

# Colour format 
def purple(word=None, font_size=None,
                    position=None, orientation=None,
                    font_path=None, random_state=None):
    h = 267 # 0 - 360
    s = 99 # 0 - 100
    l = random_state.randint(50, 80) # 0 - 100
    return "hsl({}, {}%, {}%)".format(h, s, l)


# @st.experimental_memo(persist='disk')
def create_wordcloud(df):
    # Aesthetics
    font = {'family' : 'Yahoo Sans', 'weight' : 'bold'}
    plt.rc('font', **font)

    # Create spaCy document and tokenise 
    nlp.max_length = 7000000
    doc = process_text(spacy_model, " ".join(df.query('age in @age_selection and gender in @gender_selection')['search'].to_list()))
    # doc = nlp(" ".join(df.query('age in @age_selection and gender in @gender_selection')['search'].to_list()))
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct and token.pos_ in parts_option]
    entities = [(x.lemma_, x.label_) for x in doc.ents]
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
    plt.savefig('images/wordcloud.png', dpi=1000, transparent=True)
    return fig, entities

# VENN 

@st.experimental_memo(persist='disk')
def venn_intersections(df_words):
    # Create both circles based on user selection
    left_venn = df_words.query('gender in @gender_selection1 & age in @age_selection1')
    right_venn = df_words.query('gender in @gender_selection2 & age in @age_selection2')

    # left spacy doc 
    doc1 = process_text(spacy_model, " ".join(left_venn['search'].to_list()))
    left = [token.text.capitalize() for token in doc1 if not token.is_stop and not token.is_punct and token.pos_ in parts_option]
    left = Counter(left).most_common(40)

    # right spacy doc
    doc2 = process_text(spacy_model, " ".join(right_venn['search'].to_list()))
    right = [token.text.capitalize() for token in doc2 if not token.is_stop and not token.is_punct and token.pos_ in parts_option]
    right = Counter(right).most_common(40)
    
    # create data frames 
    left = pd.DataFrame(left, columns=['search', 'count'])  
    right = pd.DataFrame(right, columns=['search', 'count'])

    return left, right


def venn_cloud(left, right):
    fig, ax = plt.subplots(figsize=(6,5), dpi=1000)

    v = venn2_wordcloud([set(left['search']), set(right['search'])], 
                        set_labels=[left_label, right_label], 
                        alpha=1,
                        ax=ax, 
                        wordcloud_kwargs=dict(font_path='fonts/Yahoo Sans-Regular.otf', prefer_horizontal=1, max_words=30, color_func=lambda *args, **kwargs: "white"))

    v.get_patch_by_id('10').set_color('#FF0080')
    v.get_patch_by_id('11').set_color('#7E1FFF')
    v.get_patch_by_id('01').set_color('#11D3CD')

    for label in v.set_labels:
        label.set_fontsize(12.)

    # Left Text
    fig_text(x=0.10, y=0.7, 
    s=f"""Words popular  
    with <{left_label}>""",
            highlight_textprops=[{'color':'#FF0080', 'weight':'bold'}],
            weight='regular', fontfamily='Yahoo Sans', fontsize=6, ha='center')

    # Right Text
    fig_text(x=0.9, y=0.7, 
    s=f"""Words popular 
    with <{right_label}>""",
            highlight_textprops=[{'color':'#11D3CD', 'weight':'bold'}],
            weight='regular', fontfamily='Yahoo Sans', fontsize=6, ha='center')

    plt.savefig('images/venn_cloud.png', dpi=1000, transparent=True)

    return fig 


# TIME SERIES

def research_dates(df):
        return (df['date'].dt.date.min().strftime('%-d %b %Y'), df['date'].dt.date.max().strftime('%-d %b %Y'))


def is_spring_summer(ds):
    dt = pd.to_datetime(ds)
    return dt.quarter == 2 | dt.quarter == 3


def holiday_cleanup(ser):
        return (ser.str.replace('[England/Wales/Northern Ireland]', '', regex=False)
                .str.replace(' [Northern Ireland]', '', regex=False)
        )

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

    # Titles
    s = '% Lift in Searches by Holiday & Event'
    s2 = 'The chart shows the <Positive> and <Negative> impact on searches during holidays' 
    fig_text(0.05, 0.92, s, fontweight='extra bold', fontsize=12, va='bottom',  color='#6001D2')
    fig_text(0.05, 0.89, s2, fontsize=6, va='bottom', highlight_textprops=[{"color": "#7E1FFF", "fontweight":'bold'},
                                                                            {"color": "grey", "fontweight":"bold"}])

    # Adjust axes positions to fit commentary
    fig.subplots_adjust(top=0.85, left=0.05)

    # Source
    fig.supxlabel(f'Source: Yahoo Internal {market} \nData covers {research_start} - {research_end}', fontsize=4, x=0.9, y=-0.05, ha='right')
    ax.get_legend().remove()
    plt.savefig('images/holiday_impact.png', dpi=1000, transparent=True)

    return fig


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
        fig_text(0.05, 0.92, s, fontweight='extra bold', fontsize=12, va='bottom',  color='#6001D2')

        fig_text(0.05, 0.89, s2, fontsize=6, va='bottom', highlight_textprops=[{"color": "#7E1FFF", "fontweight":'bold'},
                                                                                {"color": "lightgrey", "fontweight":"bold"}])

        
        # Caption
        fig.supxlabel(f'Source: Yahoo Internal - {market} \nData covers {research_start} - {research_end}', fontsize=4, x=0.9, ha='right')

        # Custom Grouping Aesthetics
        for day in range(7):
                ax.plot([day - 0.45, day + 0.45], [yticks[-1], yticks[-1]], color='black', lw=0.5)

        ax.get_legend().remove()

        # Adjust axes positions to fit commentary
        pos1 = ax.get_position() # get the original position 
        pos2 = [pos1.x0, pos1.y0 - 0.05,  pos1.width, pos1.height]
        ax.set_position(pos2)

        plt.savefig('images/weekly_impact.png', dpi=1000, transparent=True)

        return fig


# AUTOCORRELATION
@st.experimental_memo(persist='disk')
def autocorr_plot(ser, days=50):
    font = {'family' : 'Yahoo Sans', 'weight':'regular'}
    plt.rc('font', **font)

    fig, ax = plt.subplots(figsize=(6,3))
    daily_queries = df.groupby('date').size()
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
    fig_text(0.05, 0.95, s, fontweight='extra bold', fontsize=14, va='bottom', color='#6001D2')

    # Caption
    fig.supxlabel(f'Source: Yahoo Internal - {market} \nData covers {research_start} - {research_end}', fontsize=4, x=0.9, y=-0.05, ha='right')

    # Aesthetics
    for direction in ['bottom', 'left']:
        ax.spines[direction].set_lw(0.2)
        ax.spines[direction].set_color('grey')
        ax.spines[direction].set_alpha(0.5)
    sns.despine()
        
    
    plt.savefig('images/autocorr.png', dpi=1000, transparent=True)
    return fig

# -------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------


# UPLOAD
st.sidebar.markdown('# Upload data!')
st.sidebar.markdown('The csv or xls should have a column each for date, the search query, age group and gender')
uploaded_file = st.sidebar.file_uploader(label='csv only please', type='csv')

# ADD STOPWORDS
st.sidebar.markdown("""# Stopwords
Insert any stopwords below. These will be excluded from Topic Modelling & WordCloud analysis.""")
custom_stopwords = st.sidebar.text_input(label='Separate each word with a space.')
stopwords = stopwords.words('english')
stopwords.extend(custom_stopwords.split())

# LIMIT TOPICS
st.sidebar.markdown("""# Limit Topics""")
nr_topics = st.sidebar.select_slider('Limit the number of topics', 
                            options=['auto', 20, 30, 40, 50, 60, 70, 80, 100, 125, 150, 200],
                            value='auto',
                            help='Auto is the default and chooses an optimum number of topics based on the dataset. But if it generates too many, feel free to choose a lower number. ')


# Load default data
st.subheader('Welcome to the NLP tool. Upload your search data on the left or load a demo dataset to get started')

df = None
 
if uploaded_file is not None:
    st.sidebar.markdown('# Setup')
    df = pd.read_csv(uploaded_file)
    cols = df.columns.to_list()

    # Choose Correct Columns
    search = st.sidebar.selectbox('Which column are your queries in?', cols)
    date = st.sidebar.selectbox('Which column are your dates in?', cols)
    age = st.sidebar.selectbox('Which column is age in?', cols)
    gender = st.sidebar.selectbox('Which column is gender in?', cols)
    market = st.sidebar.text_input('What market should we source the slides as?')

    df = clean_df(df)
    df_questions = df.copy()
        
elif st.button('Load example dataset'):
    df = pd.read_csv('raw data/demo data.csv')
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    df_questions = df.copy()
    market = 'UK'
    

if df is not None:
    # nlp = spacy.load('en_core_web_sm')

    # Model Selection
    def format_func(option):
        return SELECTION[option]

    st.sidebar.markdown("# Language")

    SELECTION = {'en_core_web_sm':'english', 'es_core_news_sm':'spanish', 'pt_core_news_sm':'portuguese'}

    user_language = st.sidebar.selectbox('Select language for Topic Model', ['english', 'multilingual'], index=0, help='Multilingual provides support for 100+ languages, though will take a little longer to run.')
    spacy_model = st.sidebar.selectbox('Select language for Wordclouds', options=list(SELECTION.keys()), format_func=format_func, index=0)
    # spacy_model = st.sidebar.selectbox("Model name", ["en_core_web_sm", "es_core_news_sm", "pt_core_news_sm"], index=0)
    nlp = load_model(spacy_model)

    df['search'] = df['search'].apply(lambda row: ' '.join([word for word in row.split() if word not in (stopwords)]))

    # Get start, end dates and sourcing for sourcing
    research_start = df['date'].dt.date.min().strftime('%-d %b %Y')
    research_end = df['date'].dt.date.max().strftime('%-d %b %Y')
    timeframe_min = df['date'].dt.date.min()
    timeframe_max = df['date'].dt.date.min()

    with st.expander(label='Data Preview'):
        st.subheader('Preview of upload')
        st.write(df.head(10))

# DEMOGRAPHIC---------------------------------------------------------------------------------------------------
    
    st.header('Demographic Profile') 

    # Demographic Selection
    ages = df.query('age not in ["unknown", "Unknown", "Below13", "13-17"]')['age'].unique().tolist()
    genders = df.query('gender not in ["unknown", "-1", -1]')['gender'].unique().tolist()

    # User Selectable Configuration
    with st.expander(label='Configure Chart'):
        with st.form(key='demokey'):
            age_selection = st.multiselect('Select Age', ages, default=ages)
            gender_selection = st.multiselect('Select Gender', genders, default=genders)
            st.form_submit_button('Submit Choices') 

    st.pyplot(demo_barplot(df), dpi=600)

    # Download Button
    with open("images/demo_barplot.png", "rb") as file:
        btn = st.download_button(
            label="Download demo slide",
            data=file,
            file_name="demo_barplot.png",
            mime="image/png")



    # TOPIC MODELLING---------------------------------------------------------------------------------------------------
    
    st.header('Topic Model') #// TODO Topic model

    # Create topic model 
    topic_model, topics = get_topic_model(df)

    # Get list of topics and assign a new column with cleaned up topic names
    topic_list = topic_model.get_topic_info()
    topic_list = topic_list.assign(Description=topic_list['Name'].str.split("_", ).str[1:3].str.join(", ")).drop(columns=['Count'])

    with st.expander(label='Show topics'):
        st.write(topic_list.query('Topic >= 0'))

    # Plot Topic Map

    
    topic_map = get_topic_map(topic_model)
    st.plotly_chart(topic_map)


    st.header('Dendrogram') 
    
    st.markdown("""The dendrogram provides a way of seeing how the topics can be clustered together when based on **semantic similarity** - i.e they share a lot of words and themes. 
    The below explanation shows how the topic map directly translates into a dendrogram. """)
    st.image('images/dendrogram_explanation.png')
    

    # Plot Denodrogram
    yahoo_palette_strict = "#6001D2 #FFA700 #1AC567 #0F69FF #FF0080 #11D3CD #C7CDD2 #828A93".split()
    yahoo_palette_strict = "#6001D2 #907CFF #12A9FF #1AC567 #FFDE00 #FF8B12 #C7CDD2 #6001D2".split()

    fig_dendro = visualize_hierarchy(topic_model, orientation='bottom')
    st.plotly_chart(fig_dendro)


    # Plot Bar Charts 
    #st.header('Topics by Age') #// TODO Topic by age
    #st.header('Topics by Gender') #// TODO Topic by gender


    fig_barchart = topic_bars(topic_model)
    st.plotly_chart(fig_barchart)

    # Plot Topic Timeseries (FacetGrid)
    fig_topic_facet = topic_facet(df['search'], topics, df['date'])
    st.pyplot(fig_topic_facet)

    # Download Button
    with open("images/topic_time.png", "rb") as file:
        btn = st.download_button(
            label="Download slide",
            data=file,
            file_name="topic_time.png",
            mime="image/png")

    # Plot Demographic Topics
    #age_classes = df.query('age not in ["unknown", "Unknown", "Below13", "13-17"]')['age'].unique().tolist()
    #topics_by_age = topic_model.topics_per_class(df['search'], topics, df['age'].to_list())
    # topics_by_gender = topic_model.topics_per_class(df['search'], topics, gender_classes)
    # topics_by_class = pd.concat([topics_by_age, topics_by_gender])

    
    #fig_demobar = topics_demo(df, topics)
    #st.pyplot(fig)


    # SENTIMENT---------------------------------------------------------------------------------------------------

    st.header('Sentiment Analysis')
    st.markdown(
        """
    Searches with no sentiment expressed (i.e. **Neutral**) are removed from the analysis, leaving just the relative weight of positive and negative searches. 
    """)

    # Assign Sentiment
    df['score'] = df['search'].apply(sentiment_calc)
    df['sentiment'] = pd.cut(df['score'], bins=[-1, -0.2, 0.2, 1],labels=['Negative', 'Neutral', 'Positive'])

    # Date Aggregation Selection
    def format_func(option):
        return CHOICES[option]

    CHOICES = {'M':'Month', 'W-MON':'Week', 'Q':'Quarter', 'D':'Day', 'SM':'Bi-Week', 'B':'Business Day'}
    offset = st.selectbox(label='Select Date Aggregation', options=list(CHOICES.keys()), format_func=format_func)

    # Plot Sentiment
    fig_sent = sentiment_plot(df, offset=offset)
    st.pyplot(fig_sent, dpi=1000)

    # Download Button
    with open("images/sentiment.png", "rb") as file:
        btn = st.download_button(
            label="Download slide",
            data=file,
            file_name="sentiment.png",
            mime="image/png")


    # WORDCLOUD---------------------------------------------------------------------------------------------------

    st.header('Parts of Speech WordCloud')
    st.markdown("""
    Every word in the upload is analysed for it's part-of-speech, or role within the sentence. You can use the filters to pick out Nouns, Adjectives, Verbs or Pronouns, as well as layer in demographic profiles. 
    By default all searches are included, but you can specify a demographic using the filters below. 
    Any **stopwords** specified in the left hand navigation will apply to the WordCloud, so use this to remove unwanted words from the image. You will get more interesting results if you remove extremely common words. For instance, if your data is about Tesla, then the word Tesla isn't much use on it's own as it is self-evident. Adding Tesla as a stopword will upweight all of the other words made in conjunction with it, which is where the interesting insights are likely to be. 
    """)

    # Add stopwords to the default spacy dictionary
    for word in custom_stopwords.split():
        nlp.Defaults.stop_words.add(word)

    # Create copy of dataframe for use in wordclouds
    df_words = df.copy()

    # Demographic Selection
    ages = df['age'].unique().tolist()
    genders = df['gender'].unique().tolist()
    parts = ['ADJ', 'NOUN', 'VERB', 'PRON']

    # User Selectable Configuration
    with st.expander('Configure WordCloud'):
        with st.form(key='wordkey1'):
            age_selection = st.multiselect('Select Age', ages, default=ages)
            gender_selection = st.multiselect('Select Gender', genders, default=genders)
            parts_option = st.multiselect(label='Select Part of Speech', options= parts, default=['NOUN'])
            st.form_submit_button('Submit Choices') 

    # Plot Wordcloud
    fig_wordcloud, entities = create_wordcloud(df_words)
    st.pyplot(fig_wordcloud, dpi=1000)

    # Download Button
    with open("images/wordcloud.png", "rb") as file:
        btn = st.download_button(
            label="Download wordcloud",
            data=file,
            file_name="wordcloud.png",
            mime="image/png")

    # ENTITIES-----------------------------------------------------------------------------------------------
    st.header('Person & Influencer Recognition')
    st.markdown('using **Named Entity Recognition**, we try to extract the people and companies that are prominent in the queries.')
    
    entities = pd.DataFrame(entities).rename(columns={0:"entity", 1:"label"})

    people = (entities
                .query('label == "PERSON"')
                .groupby('entity').count()
                .reset_index()
                .nlargest(50, 'label')
                .drop(columns='label'))

    
    organisations = (entities
                .query('label == "ORG"')
                .groupby('entity').count()
                .reset_index()
                .nlargest(50, 'label')
                .drop(columns='label'))
    
    norp = (entities
                .query('label in ["NORP"]')
                .groupby('entity').count()
                .reset_index()
                .nlargest(50, 'label')
                .drop(columns='label'))

    # events = (entities
    #             .query('label in ["EVENT", "PRODUCT"]')
    #             .groupby('entity').count()
    #             .reset_index()
    #             .nlargest(50, 'label')
    #             .drop(columns='label'))

    entcol1, entcol2, entcol3 = st.columns(3)

    with entcol1:
        st.subheader('People')
        people
    
    with entcol2:
        st.subheader('Organisations')
        organisations

    with entcol3:
        st.subheader('National & Political')
        norp


    

    # VENN---------------------------------------------------------------------------------------------------
    st.header('Venn WordCloud')
    st.markdown("""The top 50 words used by each demographic are compared to create a venn diagram.""")
    
    # Demographic Selection
    venn_ages = df['age'].unique().tolist()
    venn_genders = df['gender'].unique().tolist()
    venn_parts = ['ADJ', 'NOUN', 'VERB', 'PRON']

    # Create a column for each venn configuration
    col1, col2 = st.columns(2)

    # Left Circle
    with col1:
        with st.expander('Configure first circle'):
            with st.form(key='venkey1'):
                age_selection1 = st.multiselect('Select Age', venn_ages, default=venn_ages)
                gender_selection1 = st.multiselect('Select Gender', venn_genders, default=['female'])
                left_label = st.text_input('Label for left circle', value='Female')
                parts_option = st.multiselect(label='Select Part of Speech', options= venn_parts, default=['NOUN'])
                st.form_submit_button('Submit Choice') 

    # Right Circle
    with col2:
        with st.expander('Configure first circle'):
            with st.form(key='venkey2'):
                age_selection2 = st.multiselect('Select Age', venn_ages, default=venn_ages)
                gender_selection2 = st.multiselect('Select Gender', venn_genders, default=['male'])
                right_label = st.text_input('Label for right circle', value='Male')
                st.form_submit_button('Submit Choice') 

    # Plot Venn Cloud
    left, right = venn_intersections(df_words)
    fig_venn = venn_cloud(left, right)
    st.pyplot(fig_venn, dpi=1000)

    # Download Button
    with open("images/venn_cloud.png", "rb") as file:
        btn = st.download_button(
            label="Download Venn Cloud slide",
            data=file,
            file_name="venn_cloud.png",
            mime="image/png")

    # ASPECT ANALYSIS
    # st.header('Aspect Analysis') 

    # aspects = []
    # for search in df['search'].tolist():
    #     doc = nlp(search)
    #     descriptive_term = ''
    #     target = ''
    #     for token in doc:
    #         if token.dep_ == 'nsubj': #and token.pos_ == 'NOUN':
    #             target = token.lemma_
    #         if token.pos_ == 'ADJ':
    #             prepend = ''
    #             for child in token.children:
    #                 if child.pos_ != 'ADV':
    #                   continue
    #                 prepend += child.text + ' '
    #             descriptive_term = prepend + token.text
    #     aspects.append({'aspect': target,
    #         'description': descriptive_term})

    # search_aspects = pd.DataFrame(aspects).query('aspect not in ["","-PRON-"] & description != ""')

    # st.dataframe(search_aspects
    #                 .groupby(['aspect', 'description'])
    #                 .agg(count=('aspect','count'))
    #                 .sort_values('count', ascending=False)
    #                 .head(50)
    #                 .reset_index()
    #                 .filter(['aspect', 'description']), width=600, height=600)


    # THE QUESTIONS---------------------------------------------------------------------------------------------------

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

    df_questions['length'] = df_questions['search'].str.len()
    df_questions['search'] = df_questions['search'].str.replace('_', ' ').str.replace(r'(co.uk|www.|http://|\d| amp|&amp|(\S{14,}))', '').str.strip().str.replace('  ', ' ').str.lower()
    st.dataframe(df_questions
            .dropna()
            .query('search.str.contains("^do |^how|what|when|why|where|^is ")', engine='python')
            .groupby('search', as_index=False)
            .size()
            .sort_values('size', ascending=False)
            .head(40)
            .filter(like='search'), width=800, height=600)


    # TIME SERIES DECOMPOSITION---------------------------------------------------------------------------------------------------

    st.header('Time Series Decomposition') # // TODO Decomposition
    st.markdown("""These plots show the lift in searches for given days of the week, public holidays and custom dates. 
    This is done by decomposing the provided data into three compontents (trend, seasonality & noise) to better understand patterns in the data. """)

    # Read in custom holidays file
    holidays = pd.read_csv('raw data/custom_holidays.csv')
    holidays['ds'] = pd.to_datetime(holidays['ds'], format='%d/%m/%Y')

    def format_func_country(option):
        return COUNTRY_CHOICES[option]

    st.subheader('Holiday and Custom Date Impact') #// TODO Holiday impact 
    COUNTRY_CHOICES = {'UK':'UnitedKingdom', 'US': 'UnitedStates', 'AU':'Australia', 'AO': 'Angola', 'AR': 'Argentina', 'AW': 'Aruba', 'AT': 'Austria', 'AZ': 'Azerbaijan', 'BD': 'Bangladesh', 'BY': 'Belarus', 'BE': 'Belgium', 'BW': 'Botswana', 'BR': 'Brazil', 'BG': 'Bulgaria', 'BI': 'Burundi', 'CA': 'Canada', 'CL': 'Chile', 'CN': 'China', 'CO': 'Colombia', 'HR': 'Croatia', 'CW': 'Curacao', 'CZ': 'Czechia', 'DK': 'Denmark', 'DJ': 'Djibouti', 'DO': 'DominicanRepublic', 'EG': 'Egypt', 'EE': 'Estonia', 'ET': 'Ethiopia', 'ECB': 'EuropeanCentralBank', 'FI': 'Finland', 'FR': 'France', 'GE': 'Georgia', 'DE': 'Germany', 'GR': 'Greece', 'HN': 'Honduras', 'HK': 'HongKong', 'HU': 'Hungary', 'IS': 'Iceland', 'IN': 'India', 'IE': 'Ireland', 'IL': 'Israel', 'IT': 'Italy', 'JM': 'Jamaica', 'JP': 'Japan', 'KZ': 'Kazakhstan', 'KE': 'Kenya', 'KR': 'Korea', 'LV': 'Latvia', 'LS': 'Lesotho', 'LT': 'Lithuania', 'LU': 'Luxembourg', 'MY': 'Malaysia', 'MW': 'Malawi', 'MX': 'Mexico', 'MA': 'Morocco', 'MZ': 'Mozambique', 'NL': 'Netherlands', 'NA': 'Namibia', 'NZ': 'NewZealand', 'NI': 'Nicaragua', 'NG': 'Nigeria', 'MK': 'NorthMacedonia', 'NO': 'Norway', 'PY': 'Paraguay', 'PE': 'Peru', 'PL': 'Poland', 'PT': 'Portugal', 'PTE': 'PortugalExt', 'RO': 'Romania', 'RU': 'Russia', 'SA': 'SaudiArabia', 'RS': 'Serbia', 'SG': 'Singapore', 'SK': 'Slovakia', 'SI': 'Slovenia', 'ZA': 'SouthAfrica', 'ES': 'Spain', 'SZ': 'Swaziland', 'SE': 'Sweden', 'CH': 'Switzerland', 'TW': 'Taiwan', 'TR': 'Turkey', 'TN': 'Tunisia', 'UA': 'Ukraine', 'AE': 'UnitedArabEmirates', 'VE': 'Venezuela', 'VN': 'Vietnam', 'ZM': 'Zambia', 'ZW': 'Zimbabwe'}

    # Interface for adding custom dates
    st.markdown("""By default, all **public holidays** in a country are used for this analysis.  But adding a custom date allows you to see the impact that it drove on searches.  
    For instance, the end of lockdown or tiered restrictions was likely a trigger for many behaviours - entering the date it happened will let you know by how much, by comparing the lift (or decline) versus the trend going into that date.""")
    with st.expander(label = 'Configure Time Series Decomposition'):
        with st.form("Enter Custom Date"): #// TODO Holidays-custom_date
            country = st.selectbox(label='Select Public Holidays to Analyse', options=list(COUNTRY_CHOICES.keys()), format_func=format_func_country)
            custom_name = st.text_input(label='Add Custom Date - Give the day you want to measure a name?', placeholder = 'Superbowl, Champions League Final, Restrictions End etc')
            custom_date = st.date_input(label='What date was it on?')
            submitted = st.form_submit_button("Submit")
            holidays = holidays.append({'holiday':custom_name, 'ds':custom_date, 'lower_window':0, 'upper_window':0}, ignore_index=True)


    #// TODO Holidays-fix US issue
    #// TODO Holidays-fix color issue

    df_prophet = (df
                    .groupby('date', as_index=False).size()
                    .rename({'date':'ds', 'size':'y'}, axis='columns')
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

    # Concat dataframes to create one which contains forecast, actuals & impact by holiday
    df_seasonality = (pd.concat([forecast[cols], df_prophet['y']], axis='columns')
                        .dropna()
                        .assign(holiday_impact=lambda df: df['holidays'] / df['y'],
                                        ss_weeklyimpact=lambda df: df['weekly_springsummer'] / df['y'],
                                        aw_weeklyimpact=lambda df: df['weekly_autumnwinter'] / df['y'],
                                        dayofweek=lambda df: df['ds'].dt.day_name(),
                                        weekday=lambda df: df['ds'].dt.dayofweek))

    df_holidays = (m.construct_holiday_dataframe(df_seasonality['ds'])
                    .assign(holiday=lambda df: holiday_cleanup(df['holiday']))
                    .set_index('ds'))


    cols = ['ds', 'holidays', 'holiday_impact', 'yhat', 'y']

    # holiday_impact = (pd.concat([df_seasonality[cols].set_index('ds'), df_holidays], axis='columns')
    #                     .reset_index()
    #                     .query('holidays != 0')
    #                     .dropna(subset=['holidays'])
    #                     .fillna(''))

    holiday_impact = df_seasonality.merge(df_holidays.reset_index(), left_on='ds', right_on='ds')

    scot = holiday_impact['holiday'].str.contains("Scotland|Boyne")

    holiday_impact = holiday_impact.loc[~scot].assign(holiday_impact=holiday_impact['holiday_impact'].clip(lower=-0.95))


    research_start, research_end = research_dates(df)

    holiday_options = holiday_impact['holiday'].to_list()
    holiday_selected = st.multiselect('Select & Deselect Holidays To Show In Chart', options=holiday_options, default=holiday_options)


    # Plot Holiday Impact
    fig_holiday = holiday_impact_plot(holiday_impact)
    st.pyplot(fig_holiday, dpi=1000)

    # Download Button
    with open("images/holiday_impact.png", "rb") as file:
        btn = st.download_button(
            label="Download holiday impact slide",
            data=file,
            file_name="holiday_impact.png",
            mime="image/png")

    st.subheader('Day of Week Impact')
    st.markdown("""Conditional seasonality is used to split **A/W** and **S/S** as behaviours tend to differ in each.  Taking the average over the course of a year hides these differences.""")

    # Plot Daily Impact
    fig_weekly = weekly_impact_plot(df_seasonality)
    st.pyplot(fig_weekly, dpi=1000)

    # Download Button
    with open("images/weekly_impact.png", "rb") as file:
        btn = st.download_button(
            label="Download weekly impact slide",
            data=file,
            file_name="weekly_impact.png",
            mime="image/png")


    # AUTOCORRELATION---------------------------------------------------------------------------------------------------

    st.header('Autocorrelation') #//  Autocorrelation
    st.markdown("""These plots graphically summarise the strength of a relationship between any given day and that of a day at prior time steps (lags). An autocorrelation of **1** represents a perfect positive correlation, while **-1** represents a perfect negative correlation.
    The purple shaded area is the 95% confidence interval - anything within here shows no significant correlation.""")

    # Plot Autocorrelation
    days = st.slider(label='Number of days', min_value=10, max_value=370, step=10, value=50, format='%d', help='mailto:jamie.nathan@yahooinc')
    fig_autocorr = autocorr_plot(df['date'], days=days)
    st.pyplot(fig_autocorr, dpi=1000)

    # Download Button
    with open("images/autocorr.png", "rb") as file:
        btn = st.download_button(
            label="Download autocorrelation slide",
            data=file,
            file_name="autocorr.png",
            mime="image/png")
