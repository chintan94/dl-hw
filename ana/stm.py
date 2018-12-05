import gensim
from gensim import corpora,models
from gensim.models import LdaModel
from gensim.test.utils import datapath
from gensim.parsing.preprocessing import remove_stopwords,strip_punctuation, strip_numeric,strip_short, stem_text
import pandas as pd
import unidecode
import csv
import datetime as dt

from gensim.models import ldaseqmodel

import pyLDAvis
import pyLDAvis.gensim  # don't skip this

import matplotlib.pyplot as plt

def preprocess(doc):
    return strip_short(remove_stopwords(strip_numeric(strip_punctuation(doc.lower()))),3).split()

def lda_modeling_df(df):
    corpus, dictionary = prep_corpus(df)
    lda_model = LdaModel(corpus=corpus,  # This code runs your lda
                             id2word=dictionary, 
                             random_state=100, 
                             num_topics=15,
                             passes=5,
                             chunksize=10000,
                             alpha='asymmetric',
                             decay=0.5,
                             offset=64,
                             eta=None,
                             eval_every=0,
                             iterations=100,
                             gamma_threshold=0.001,
                             per_word_topics=True)
    return lda_model, corpus, dictionary

def prep_corpus(df):
    tags = [tag for tag in df["review_body"]]
    corpus = [preprocess(tag) for tag in tags]
    dictionary = corpora.Dictionary(corpus)
    corpus = [dictionary.doc2bow(preprocess(tag)) for tag in tags]
    dictionary.filter_extremes(no_below=10, no_above=0.8)
    corpus = [dictionary.doc2bow(preprocess(tag)) for tag in tags]
    return corpus, dictionary
    
def print_lda_models(lda_model, dictionary):
    for i in range(15):
        words = lda_model.get_topic_terms(i, topn=10)
        print("Topic : " + str(i))
        for i in words:
            print("Word: " + str(dictionary[i[0]]) + "\t\t Weight: " + str(i[1])) 
        print("\n")

filename = "amazon_reviews_us_Beauty_v1_00.tsv"
url = 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Beauty_v1_00.tsv.gz'

a = [] 
with open(filename) as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t' )
    for row in reader:
        a.append(row)
        break

fdf = pd.read_csv(filename, delimiter='\t', names=a[0], skiprows=[0])
mask = fdf.isnull().sum(axis=1) != 7
fdf = fdf[mask]
fdf["review_date"] = pd.to_datetime(fdf['review_date'],format = "%Y-%m-%d",errors='coerce')
fdf.dropna(inplace=True)
fdf["year"] = fdf["review_date"].map(lambda x: x.year)

year_count = fdf.groupby('year')['review_id'].nunique()

years = list(year_count.keys())
years.remove(2000)
df_list = []
thresh = 10000
for year in years:
    temp_df = fdf[fdf["year"] == year]
    if (year_count[year] > thresh):
        temp_df = temp_df.sample(thresh)
    df_list.append(temp_df)    


concat_dfs = []
for i in range(5):
    concat_dfs.append(pd.concat(df_list[3 * i: 3 * i + 3]))
time_slice = [len(i) for i in df_list]


lda_model, corpus, dictionary = lda_modeling_df(concat_dfs[0])
print("Partition 1")
print("Dictionary length : " + str(len(dictionary)))
print_lda_models(lda_model, dictionary)

lda_model.log_perplexity(corpus)
vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
temp_file = datapath("lda_model1")
lda_model.save(temp_file)

#pyLDAvis.display(vis)
pyLDAvis.save_html(vis, 'lda1.html')


lda_model2, corpus2, dictionary2 = lda_modeling_df(concat_dfs[1])

print("Partition 2")
print("Dictionary length : " + str(len(dictionary2)))
print_lda_models(lda_model2, dictionary2)

lda_model2.log_perplexity(corpus2)
vis2 = pyLDAvis.gensim.prepare(lda_model2, corpus2, dictionary2)

temp_file = datapath("lda_model2")
lda_model2.save(temp_file)

#pyLDAvis.display(vis2)
pyLDAvis.save_html(vis2, 'lda2.html')



lda_model3, corpus3, dictionary3 = lda_modeling_df(concat_dfs[2])

print("Partition 3")
print("Dictionary length : " + str(len(dictionary3)))
print_lda_models(lda_model3, dictionary3)

lda_model3.log_perplexity(corpus3)
vis3 = pyLDAvis.gensim.prepare(lda_model3, corpus3, dictionary3)


temp_file = datapath("lda_model3")
lda_model3.save(temp_file)

#pyLDAvis.display(vis3)
pyLDAvis.save_html(vis3, 'lda3.html')


lda_model4, corpus4, dictionary4 = lda_modeling_df(concat_dfs[3])

print("Partition 4")
print("Dictionary length : " + str(len(dictionary4)))
print_lda_models(lda_model4, dictionary4)

lda_model4.log_perplexity(corpus4)
vis4 = pyLDAvis.gensim.prepare(lda_model4, corpus4, dictionary4)


temp_file = datapath("lda_model4")
lda_model4.save(temp_file)

#pyLDAvis.display(vis4)
pyLDAvis.save_html(vis4, 'lda4.html')


lda_model5, corpus5, dictionary5 = lda_modeling_df(concat_dfs[4])

print("Partition 5")
print("Dictionary length : " + str(len(dictionary5)))
print_lda_models(lda_model5, dictionary5)

lda_model5.log_perplexity(corpus5)
vis5 = pyLDAvis.gensim.prepare(lda_model5, corpus5, dictionary5)


temp_file = datapath("lda_model5")
lda_model5.save(temp_file)

#pyLDAvis.display(vis5)
pyLDAvis.save_html(vis5, 'lda5.html')

flda_model, fcorpus, fdictionary = lda_modeling_df(pd.concat(concat_dfs))

print("Complete Partition")
print("Dictionary length : " + str(len(fdictionary)))
print_lda_models(flda_model, fdictionary)

flda_model.log_perplexity(fcorpus)
fvis = pyLDAvis.gensim.prepare(flda_model, fcorpus, fdictionary)


temp_file = datapath("flda_model")
flda_model.save(temp_file)

#pyLDAvis.display(fvis)
pyLDAvis.save_html(fvis, 'flda.html')

