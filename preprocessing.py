import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import seaborn as sns

from nltk.corpus import stopwords 
from nltk import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from scipy.misc import imread
from wordcloud import WordCloud, STOPWORDS
import random

# global variables
# Stemmatizer and lemmatizer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# visually understand data
def show_dataframe(df):
	count_srs = df['type'].value_counts()
	plt.figure(figsize=(12,4))
	sns.barplot(count_srs.index, count_srs.values, palette='RdBu', alpha=1)
	plt.ylabel('Number of Datapoints', fontsize=12)
	plt.xlabel('Types', fontsize=12)
	plt.show()

# get num_words per row
def num_words(row):
	len(x.split())

# create new dataframe with only "thinking" and "feeling", and "judging" and "perceiving"
def dataframe_by_dichotomy(index):
    new_df = dataframe.copy()
    new_df['type'] = new_df['type'].apply(lambda x: x[index])
    return new_df

# clean given block of text
def clean_text(txt):
    # Cache the stop words for speed 
    cachedStopWords = stopwords.words("english")
    # Remove urls
    txt = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'link', txt)
    # Keep only words
    txt = re.sub("[^a-zA-Z]", " ", txt)
    # Remove {enfp, entp, infp...}
    
    # Remove spaces > 1 and making everything lowercase
    txt = txt.lower()
    txt = re.sub('[ei]+[ns]+[ft]+[jp]', ' ', txt)
    txt = re.sub(' +', ' ', txt)
    txt = lemmatizer.lemmatize(" ".join([w for w in txt.split(' ') if w not in cachedStopWords]))
    return txt

def clean_data(df):
    new_df = df.copy()
    new_df['posts'] = new_df['posts'].apply(lambda x: clean_text(x))
    return new_df

# trial = dataframe['posts'][2]
# print("\nBefore preprocessing:\n\n", trial[0:500])
# print("\nAfter preprocessing:\n\n", clean_text(trial)[0:500])

# clean_dataframe = clean_data(dataframe)
# clean_dataframe.to_csv('mtbi_clean.csv')



def createVectorizedCSV():
    
    tfidfVect = TfidfVectorizer(analyzer="word", 
                                 max_features=1000, 
                                 tokenizer=None,    
                                 preprocessor=None, 
                                 stop_words=None,  
                                 ngram_range=(1,1))
                               #  max_df=0.5)
                               #  min_df=0.1
    
    new_features = tfidfVect.fit_transform(clean_dataframe_think['posts'])

    output_dataframe = pd.DataFrame(new_features.toarray(), columns=tfidfVect.get_feature_names())

    output_dataframe['type'] = clean_dataframe_think['type']
    output_dataframe.to_csv('mtbi_bow_thinking_1000_features.csv', index=False)

def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl(240, 100%%, %d%%)" % random.randint(60, 100)

def make_word_cloud():
    fig, ax = plt.subplots(1, sharex=True, figsize=(30,10))
    df = clean_data(df_think)
    wordcloud = WordCloud().generate(df['posts'].to_string())
    ax.imshow(wordcloud.recolor(color_func=color_func, random_state=3),
        interpolation="bilinear")
    ax.axis("off")
    plt.show()

def findNumWords(word):
    
    thinkWords = clean_dataframe_think[clean_dataframe_think['type'] == 'T']['posts'].str.count(word).sum()
    feelWords = clean_dataframe_think[clean_dataframe_think['type'] == 'F']['posts'].str.count(word).sum()
    return [thinkWords, feelWords]

def totalLength():
    thinkWords = clean_dataframe_think[clean_dataframe_think['type'] == 'T']['posts'].str.len().sum()
    feelWords = clean_dataframe_think[clean_dataframe_think['type'] == 'F']['posts'].str.len().sum()
    return [thinkWords, feelWords]

def printImpWords(): 
    print("\nTotal words for 'thinking' and 'feeling' are %s and %s\n" % (totalLength()[0] , totalLength()[1]))
    print("\nThe following words occur so many times in dataset with 'thinking' and 'feeling' type respectively:")
    print("'feel': %d and %d"   % (findNumWords("feel")[0], findNumWords("feel")[1]))
    print("'love': %d and %d"   % (findNumWords("love")[0], findNumWords("love")[1]))
    print("'feeling': %d and %d"   % (findNumWords("feeling")[0], findNumWords("feeling")[1]))
    print("'happy': %d and %d"   % (findNumWords("happy")[0], findNumWords("happy")[1]))
    print("'beautiful': %d and %d"   % (findNumWords("beautiful")[0], findNumWords("beautiful")[1]))
    print("'use': %d and %d"   % (findNumWords("use")[0], findNumWords("use")[1]))
    print("'really': %d and %d"   % (findNumWords("really")[0], findNumWords("really")[1]))
    print("'heart': %d and %d"   % (findNumWords("heart")[0], findNumWords("heart")[1]))
    print("'thank': %d and %d"   % (findNumWords("thank")[0], findNumWords("thank")[1]))
    print("'felt': %d and %d"   % (findNumWords("felt")[0], findNumWords("felt")[1]))
    print("'shit': %d and %d"   % (findNumWords("shit")[0], findNumWords("shit")[1]))
    print("'hope': %d and %d"   % (findNumWords("hope")[0], findNumWords("hope")[1]))
    print("'sad': %d and %d"   % (findNumWords("sad")[0], findNumWords("sad")[1]))
    print("'deep': %d and %d"   % (findNumWords("deep")[0], findNumWords("deep")[1]))
    print("'problem': %d and %d"   % (findNumWords("problem")[0], findNumWords("problem")[1]))
    print("'information': %d and %d"   % (findNumWords("information")[0], findNumWords("information")[1]))
    print("'science': %d and %d"   % (findNumWords("science")[0], findNumWords("science")[1]))
    print("'logic': %d and %d"   % (findNumWords("logic")[0], findNumWords("logic")[1]))
    print("'haha': %d and %d"   % (findNumWords("haha")[0], findNumWords("haha")[1]))
    print("'knowledge': %d and %d"   % (findNumWords("knowledge")[0], findNumWords("knowledge")[1]))
    print("'argument': %d and %d"   % (findNumWords("argument")[0], findNumWords("argument")[1]))
    print("'song': %d and %d"   % (findNumWords("song")[0], findNumWords("song")[1]))
    print("\n")

### read data
dataframe = pd.read_csv('mbti_1.csv')

### create new dataframes specifically for "thinking" and "judging" dicotomies
df_think = dataframe_by_dichotomy(2)
# df_judging = dataframe_by_dichotomy(3)

### things to print/show
# print(dataframe['posts'][2])
# print(dataframe.shape)
# show_dataframe(dataframe)
# show_dataframe(df_think)
# show_dataframe(df_judging)
make_word_cloud()
clean_dataframe_think = clean_data(df_think)
createVectorizedCSV()
printImpWords()
