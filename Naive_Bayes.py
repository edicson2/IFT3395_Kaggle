# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy import sparse
import string


# Erase all the punctuation
def pd_translate(df):
    df = df.replace(r'\n',' ', regex=True)
    punct = string.punctuation
    transtab = str.maketrans(dict.fromkeys(punct, ' '))
    return df.assign(
        Abstract='¿'.join(df['Abstract'].tolist()).translate(transtab).split('¿')
    )


def load_common_words(file):
    common_words = pd.read_csv(file,header=None)
    common_words = common_words[0].str.split(' ')
    return np.array(common_words[0])


def pre_traitement(df, common_words):
    # Remove the punctuation, blank spaces and put everything in lowercase
    df = df.assign(Abstract=pd_translate(df)['Abstract'].str.lower())
    df = df.assign(Abstract=df['Abstract'].str.strip())
    df = df.replace(r'\s+',' ', regex=True)
    
    # Split each phrase in words
    df = df.assign(Abstract=df['Abstract'].str.split(' '))

    abstracts = []

    for article in df['Abstract'] :
        article = np.array(article)
        index = []
        for common_word in common_words :
            index = np.where(article == common_word)
            article = np.delete(article, index)
        abstracts.append(article)

    df = df.assign(Abstract=list(abstracts))

    return df


def create_category_maps(data):
    words_par_category = {}
    # Create an entry for each unique category
    for category in data['Category'].unique():
        df_cat = data.loc[data['Category'] == category]
        
        total_class_words = np.array([])
        
        # Add the word to a list for the category
        for row in df_cat['Abstract']:
            total_class_words = np.concatenate((total_class_words, row), axis=None)
        
        # Count the number of times a word is repeated in a category
        words_by_class = {}
        for (i, elem) in enumerate(total_class_words):
            if elem in words_by_class:
                words_by_class[elem] += 1
            else:
                words_by_class[elem] = 1
        
        words_par_category[category] = words_by_class
        
    return words_par_category


# Return the size of the vocabulary
def taille_vocabulaire(mots_par_category):
    total_size = 0
    for category in mots_par_category.items():
        total_size += len(category[1])
    return total_size



df= pd.read_csv("train.csv")
common_words = load_common_words('common_english_words.txt')
unique_labels = df['Category'].unique()
data = pre_traitement(df, common_words)
category_maps = create_category_maps(data)

#print(common_words)
#print(category_maps['astro-ph'])

def remove_common_words(df, common_list) :
    for row in df['Abstract'] :
        print(type(row))

print(data)
#remove_common_words(df, [0,1])
