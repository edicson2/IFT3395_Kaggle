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

# Count the number of words in a list and erase the ones that are in the another list
def count_and_erase_words(lst, common_list):

    # Count the ocurrence of each word
    word_count = {}
    for elem in lst:
        if elem in word_count:
            word_count[elem] += 1
        else:
            word_count[elem] = 1

    # Delete the 'common' words from the list
    if common_list.size != 0:
        for item in common_list:
            if item in word_count:
                word_count.pop(item, None)

    # Calculate the probability for each word
    #nom_elements = len(lst)
    #for key in word_count:
    #    word_count[key] = word_count[key] / nom_elements
    
    return list(word_count.keys())


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

    # add a column with the probability for every word
    prob = []
    for row in df['Abstract']:
        prob.append(count_and_erase_words(row, common_words))
    #df['probability'] = prob
    df = df.assign(Abstract=prob)
    
    return df


def create_maps_by_category(data):
    mots_par_category = {}
    # Create an entry for each unique category
    for category in data['Category'].unique():
        df_cat = data.loc[data['Category'] == category]
        total_mots_par_classe = []
        
        # Add the word to a list for the category
        for row in df_cat['Abstract']:
            total_mots_par_classe += row
        
        # Count the number of times a word is repeated in a category
        words_by_class = {}
        for (i, elem) in enumerate(total_mots_par_classe):
            if elem in words_by_class:
                words_by_class[elem] += 1
            else:
                words_by_class[elem] = 1
        mots_par_category[category] = words_by_class
        
    return mots_par_category


def taille_vocabulaire(mots_par_category):
    total_size = 0
    for category in mots_par_category.items():
        total_size += len(category[1])
    return total_size


df= pd.read_csv("train.csv")
common_words = load_common_words('common_english_words.txt')
unique_labels = df['Category'].unique()
data = pre_traitement(df, common_words)