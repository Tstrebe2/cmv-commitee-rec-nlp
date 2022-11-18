# Databricks notebook source
import numpy as np
import pandas as pd
import spacy
import en_core_web_sm
import torch
import tensorflow_hub as hub
from sklearn.manifold import TSNE
from symspellpy import SymSpell, Verbosity
import re

# from summarizer import Summarizer, TransformerSummarizer
nlp = spacy.load("en_core_web_sm")
nlp.Defaults.stop_words |= {"year","month","week","number","veteran","va","committee","program","minority","report", 'dear'}
local_stop_words = {"year","month","week","number","veteran","veterans","va","committee","program","minority","report",'secretary', 'member', 'subcommittee', 'department', 'meeting', 'information', 'service', 'director', 'provide', 'center', 'concern', 'affair', 'advisory', 'need', 'american', 'meet', 'establish', 'organization','second','annual','government',
'include', 'annex', 'work', 'plan', 'issue', 'serve', 'health', 'review', 'group', 'support', 'specific', 'care', 'office', 'page', 'day', 'continue', 'factor', 'dear',
                           'recommend', 'recommendation', 'process', 'recommendation'}

# local_stop_words = local_stop_words.lower()
nlp.Defaults.stop_words |= local_stop_words



# sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
# # dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
# # sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)



# COMMAND ----------


import pickle
with open('/dbfs/NAII/CMV Reports/stopwords_by_year', 'rb') as fp:
    stopByYear = pickle.load(fp)

# COMMAND ----------

allrecomm = spark.sql('select * from NAII.CMV_Reports_Article_Recommendations_lists order by Year').toPandas()

# COMMAND ----------

def lemmatization(reports):
    # Tags I want to remove from the text
    removal= ['ADV','PRON','CCONJ','PUNCT','PART','DET','ADP','SPACE', 'NUM', 'SYM']
    stops = nlp.Defaults.stop_words
    tokens = []
    for summary in nlp.pipe(reports):
        proj_tok = [token.lemma_.lower() for token in summary if token.pos_ not in removal  and token.is_alpha]
#         proj_tok = [sym_spell.lookup(t, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True, ignore_token=r"\w+\d")[0].term for t in proj_tok]
        proj_tok = [w for w in proj_tok if w not in stops]
        tokens.append(proj_tok)
    return tokens

# COMMAND ----------

allrecomm['Recommlst'] = [i.split("&&") for i in allrecomm['Recommendations']]
allrecomm['RecommendationsClean'] = [i.replace("&&", '''.\n''') for i in allrecomm['Recommendations']]
allrecomm['Recommlststr'] = [re.sub(r'\d\. ',r'',re.sub(r'\dd\. ',r'',i)).replace("&&", '''.\n''') for i in allrecomm['Recommendations']]
# allrecomm['Recommlststr'] = allrecomm.Recommendations.apply(lambda x: '''.\n'''.join([str(i) for i in x]))  

# COMMAND ----------

display(allrecomm)

# COMMAND ----------

allrecommdelta = allrecomm.copy()

# COMMAND ----------

# pip install bitarray

# COMMAND ----------

bart = torch.hub.load('pytorch/fairseq', 'bart.large.cnn') 

# COMMAND ----------

allrecomm['RecommendationsClean'].loc[0]
# def data_clean(text):
    

# COMMAND ----------

import string
# bart = torch.hub.load('pytorch/fairseq', 'bart.large.cnn') 
def allsummary(body):
#     body = ''.join([word  for word in body.strip() if word not in (nlp.Defaults.stop_words)])
    body = ''.join([word  for word in body.strip() if word not in (local_stop_words)])
#     body = ''.join([word for word in body.strip() if word not in string.punctuation])
    body = body.lower()
    
    
#     summarystr = str(bart.sample(body, beam=4, lenpen=2.0, max_len_b=240, min_len=100, no_repeat_ngram_size=5))
    summarystr = str(bart.sample(body, beam=4, min_len=100, no_repeat_ngram_size=5))
    return summarystr



allrecommdelta = allrecomm.copy()
allrecommdelta = allrecommdelta.sort_values(by=['Year'])
allrecommdelta = allrecommdelta[['Year','Recommendations','Recommlststr','RecommendationsClean']]
allrecommdelta['allsummary'] = allrecommdelta.Recommlststr.apply(allsummary)

allrecommdelta['originalstrlen'] = allrecommdelta['Recommlststr'].str.split().apply(len).value_counts()
allrecommdelta['summrstrlen'] = allrecommdelta['allsummary'].str.split().apply(len).value_counts()

allrecommdelta_disp = allrecommdelta[['Year','RecommendationsClean','allsummary','originalstrlen','summrstrlen']]

display(allrecommdelta[['Year','RecommendationsClean','allsummary','originalstrlen','summrstrlen']])

# COMMAND ----------

allrecommdelta['originalstrlen'] = allrecommdelta['Recommlststr'].str.split().apply(len)
allrecommdelta['summrstrlen'] = allrecommdelta['allsummary'].str.split().apply(len)
display(allrecommdelta)

# COMMAND ----------

allrecommdelta_cp = allrecommdelta.copy()

# COMMAND ----------

def lemmatization(reports, year = None):
    # Tags I want to remove from the text
    removal= ['ADV','PRON','CCONJ','PUNCT','PART','DET','ADP','SPACE', 'NUM', 'SYM']
    
    
    nlp.Defaults.stop_words |= local_stop_words
    if year is not None:
        nlp.Defaults.stop_words |= set(stopByYear[year])
    stops = nlp.Defaults.stop_words
    
    tokens = []
    for summary in nlp.pipe(reports):
#     for summary in reports.split(): 
        proj_tok = [token.lemma_.lower() for token in summary if token.pos_ not in removal  and token.is_alpha]
        proj_tok = [sym_spell.lookup(t, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True, ignore_token=r"\w+\d")[0].term for t in proj_tok]
        proj_tok = [w for w in proj_tok if w not in stops]
        tokens.append(proj_tok)
    return tokens

# COMMAND ----------

# allrecommdelta.loc[0].model.visualize_topics()

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import string


# sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
# embeddings = sentence_model.encode(docs, show_progress_bar=True)

model = []
topics = []
probs = []
for i in range(len(allrecommdelta)):
    topic_model = BERTopic(embedding_model='all-MiniLM-L6-v2',
#                            calculate_probabilities=True,
                           low_memory=True,
    #                        replace=True,
                           verbose=False,
#                            min_topic_size = 11,
#                            nr_topics=11
                          )  
#     paragraph_txt = lemmatization(nlp(str(allrecommdelta.loc[i]['RecommendationsClean'])))
    text = allrecommdelta.loc[i]['RecommendationsClean'].lower()
    paragraph_txt = [word.strip() for word in text.split() if len(word) > 5]
    paragraph_txt = [word for word in paragraph_txt if len(word) > 5]
    paragraph_txt = [word for word in paragraph_txt if word not in nlp.Defaults.stop_words]
#     paragraph_txt = [word1 for word1 in paragraph_txt if word1 not in local_stop_words]
# #     topic,probs = topic_model.fit_transform(paragraph_txt, embeddings)
    topic,prob = topic_model.fit_transform(paragraph_txt)
    model.append(topic_model)
    topics.append([])
    topics[i].append(topic)
    probs.append([])
    probs[i].append(prob)
allrecommdelta['model'] = model
allrecommdelta['topics'] = topics
allrecommdelta['probs'] = probs
# fig = topic_model.visualize_topics()
allrecommdelta.set_index('Year', inplace=True)

# COMMAND ----------

allrecommdelta.head(len(allrecommdelta))


# COMMAND ----------

uniqueyears = allrecommdelta.index.sort_values().unique()

# COMMAND ----------

uniqueyears

# COMMAND ----------

# visualize_topics()

# visualize_heatmap()
# allrecommdelta.loc[1996].model.visualize_distribution(allrecommdelta.loc[1996].probs[len(allrecommdelta.loc[1996].probs) - 1], min_probability = 0.001)
allrecommdelta.loc[2007].model.visualize_barchart()

# COMMAND ----------

allrecommdelta.loc[2008].model.visualize_barchart()

# COMMAND ----------

len(allrecommdelta)

# COMMAND ----------

listoftext = {}
for i in range(len(allrecommdelta)):
#     listoftext = listoftext | allrecommdelta.loc[i].model.get_topics()
    print('*********************************Year:-',allrecommdelta.index)
    print (allrecommdelta.iloc[i].model.get_topics())

# COMMAND ----------

# String similarity and cosine similarity 

# COMMAND ----------

import string
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.feature_extraction.text import CountVectorizer
import pkg_resources
# from nltk.corpus import stopwords
# # stopwords = stopwords.words('english')

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# COMMAND ----------

def data_clean(text):
    text = ''.join([word for word in text.strip() if word not in string.punctuation])
    text = text.lower()
    text = ''.join([word  for word in text.strip() if word not in (nlp.Defaults.stop_words)])
#     text = lemmatization([text]))
  
#     text = [word.lemma_.lower() for word in text.strip() if word.pos_ not in removal  and word.is_alpha]
#     text = ''.join([sym_spell.lookup(t, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True, ignore_token=r"\w+\d")[0].term for t in text.split()])
#         proj_tok = [w for w in proj_tok if w not in stops]
    
    
    return text
sentences = allrecommdelta.RecommendationsClean.tolist()
cleaned_sentences = list(map(data_clean,sentences))
cleaned_sentences

# COMMAND ----------

# vector = CountVectorizer().fit_transform(cleaned_sentences)
# vectors = vector.toarray()
# vectors




from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.feature_extraction import text
stops = nlp.Defaults.stop_words
stops  |= local_stop_words
tv_data = TfidfVectorizer(stop_words=stops, ngram_range = (1,1), max_df = .8, min_df = .01)
data_tv = tv_data.fit_transform(cleaned_sentences)
costheeta = cosine_similarity(data_tv,data_tv )
costheetadf = pd.DataFrame(costheeta)
costheetadf.columns = allrecommdelta.index
costheetadf.index = allrecommdelta.index
import seaborn as sns; sns.set_theme()
sns.set(rc = {'figure.figsize':(25,10)})
sns.heatmap(costheetadf, vmin=0, vmax=1, cmap="YlGnBu")

# COMMAND ----------

print(tv_data)

# COMMAND ----------

allrecommdelta.loc[1996].model.get_topics()

# COMMAND ----------

allrecommdelta.loc[2017].model.get_topics()

# COMMAND ----------

allrecommdelta.loc[2018].model.get_topics()

# COMMAND ----------

