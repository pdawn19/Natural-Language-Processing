import warnings
import tempfile
from spacy.language import EntityRecognizer
import spacy
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import gensim
from gensim import corpora, models, similarities
from string import punctuation
from sklearn.feature_extraction import stop_words
from nltk.corpus import stopwords, words
from datetime import datetime, timedelta
import os
import sys
import numpy as np
import json
import re
# import matplotlib.pyplot as plt
import pandas as pd

# NLTK Stop words library
import nltk
nltk.download('stopwords')
stop_words = stopwords.words('english')
stop_words.extend(['from', '<3', 'subject', 're', 'edu', 'use', 'nan', 'be', 'would',
                   'will', 'can', 'shall', 'https', 'could', "be", "#", "dem", "dat", "http", "www"])

# spacy library
# from nltk import word_tokenize, pos_tag
spacy_nlp = spacy.load('en_core_web_lg')
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
TEMP_FOLDER = tempfile.gettempdir()
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------------------------------------
# Disable print to console


def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore print to console


def enablePrint():
    sys.stdout = sys.__stdout__
# -----------------------------------------------------------------------------------------------------------


'''Functions for categorizing: Return Dataframe and the model'''


def test_eta(eta, dictionary, ntopics, texts, data, print_topics=True, print_dist=True):

    np.random.seed(42)  # set the random seed for repeatability
    # get the bow-format lines with the set dictionary
    bow = [dictionary.doc2bow(line) for line in texts]
    blockPrint()
    with (np.errstate(divide='ignore')):  # ignore divide-by-zero warnings
        model = gensim.models.ldamodel.LdaModel(
            corpus=bow, id2word=dictionary, num_topics=ntopics,
            random_state=42, chunksize=700, eta=eta,
            eval_every=-1, alpha='auto', update_every=1,
            passes=150, per_word_topics=True)
    if print_topics:
        topic_dict = {}
        # display the top terms for each topic
        for topic in range(ntopics):
            topic_dict[str(topic)] = [dictionary[w]
                                      for w, p in model.get_topic_terms(topic, topn=6)]
    if print_dist:
        # display the topic probabilities for each document
        line_list, probability, topics_list = [], [], []
        for line, bag in zip(data, bow):
            max1 = -1
            for topic, prob in model.get_document_topics(bag):
                if prob*100 > max1:
                    max1 = prob * 100
                    max_topic = topic
            if max1 < 20:
                max_topic = 7
            topics_list.append(max_topic)
            line_list.append(line)
            probability.append(float(max1))
        df_topics = pd.DataFrame(
            columns=['documents', 'topics', 'probability'])
        df_topics['documents'] = line_list
        df_topics['topics'] = topics_list
        df_topics['probability'] = probability
        enablePrint()
    return model, df_topics, topic_dict

# -----------------------------------------------------------------------------------------------------------


'''Train on the data '''


def create_eta(priors, etadict, ntopics):
    # create a (ntopics, nterms) matrix and fill with 1
    eta = np.full(shape=(ntopics, len(etadict)), fill_value=1)
    for word, topic in priors.items():  # for each word in the list of priors
        # look up the word in the dictionary
        keyindex = [index for index, term in etadict.items() if term == word]
        if (len(keyindex) > 0):  # if it's in the dictionary
            eta[topic, keyindex[0]] = 1e7  # put a large number in there
    # normalize so that the probabilities sum to 1 over all topics
    eta = np.divide(eta, eta.sum(axis=0))
    return eta

# -----------------------------------------------------------------------------------------------------------


'''
    Data preprocessing: Write a function to perform the pre processing steps on the entire dataset
'''


def stemming_and_lemmatizing(text):
    return WordNetLemmatizer().lemmatize(text)


# -----------------------------------------------------------------------------------------------------------

''' Tokenize and lemmatize '''


def preprocessing(text):
    result = []
    for token in gensim.utils.simple_preprocess(text, deacc=True):
        if token not in gensim.parsing.preprocessing.STOPWORDS and token not in list(spacy_stopwords) and token not in list(stop_words) and len(token) > 3:
            result.append(stemming_and_lemmatizing(token))
    return result

# -----------------------------------------------------------------------------------------------------------


''' Main function for getting the topic model chart:'''


def topic_models(allposts, groups):

    if allposts == None and groups == None:
        return {'ids': [], 'parents': [], 'labels': []}
    data = []
    days_before = (datetime.today()-timedelta(days=180)).isoformat()
    for post in allposts:
        data.append(post["message"])
        if post['created_time'] >= days_before:
            data.append(post["message"])
    for group in groups:
        data.append(group["name"])

    ''' Data Cleaning:'''
    if len(data)>100:
        data = data[:100]
    # Remove Emails
    data = [re.sub(r'\S*@\S*\s?', ' ', sent) for sent in data]

    # Remove new line characters
    data = [re.sub(r'\s+', ' ', sent) for sent in data]

    # Remove distracting single quotes
    data = [re.sub("\'", " ", sent) for sent in data]
    data = [re.sub("\,", " ", sent) for sent in data]
    data = [re.sub("\.", " ", sent) for sent in data]
    data = [re.sub("\!", " ", sent) for sent in data]
    data = [re.sub("\?", " ", sent) for sent in data]
    data = [re.sub("\-", " ", sent) for sent in data]
    data = [re.sub("\"", " ", sent) for sent in data]
    data = [re.sub("\#", " ", sent) for sent in data]
    data = [re.sub("\<3", " ", sent) for sent in data]
    data = [re.sub("\_", " ", sent) for sent in data]
    data = [re.sub("\-", " ", sent) for sent in data]

    '''Data Preprocessing'''
    doc_list = []
    for doc in data:
        doc_list.append(preprocessing(doc))

    # remove common words and tokenize
    list1 = ['RT', 'rt', '']
    stop_words = nltk.corpus.stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'nan', 'be', 'would',
                       'will', 'can', 'shall', 'https', 'could', "be", "#", "dem", "dat", "greeeeen"])
    stoplist = stop_words + list(punctuation) + list1

    texts = [[word.lower() for word in document if word not in stoplist]
             for document in doc_list]
    dictionary = corpora.Dictionary(texts)
    # store the dictionary, for future reference
    dictionary.save(os.path.join(TEMP_FOLDER, 'elon.dict'))

    # travelling: wanderlust,
    '''give seed words'''
    apriori_harder = {
        'travel': 0, 'mountain': 0, 'place': 0, 'beach': 0, 'hill': 0, 'sun': 0, 'ride': 0, 'car': 0, 'hike': 0, 'hotel': 0, 'arizona': 0, 'tempe': 0, 'water': 0, 'climb': 0, 'journey': 0, 'passerby': 0, 'holiday': 0, 'USA': 0, 'america': 0, 'Everest': 0, 'nature': 0, 'trek': 0, 'walk': 0, 'terrain': 0, 'traveller': 0, 'destination': 0, 'adventure': 0, 'trip': 0, 'itinerary': 0, 'naturelover': 0, 'peaceful': 0, 'beautiful': 0, 'routes': 0, 'restaurant': 0, 'resort': 0, 'picnic': 0, 'greenery': 0, 'plants': 0, 'food': 0, 'energized': 0, 'breathtaking': 0, 'view': 0, 'Mountaineering': 0, 'Hiking': 0, 'Biking': 0, 'Driving': 0, 'trekking': 0, 'diving': 0, 'rome': 0,
        'game': 1, 'team': 1, 'win': 1, 'play': 1, 'season': 1, 'second': 1, 'victory': 1, 'sports': 1, 'cycle': 1, 'archer': 1, 'fence': 1, 'goal': 1, 'ball': 1, 'triathlon': 1, 'basketball': 1, 'baseball': 1, 'football': 1, 'base': 1, 'love': 1, 'passion': 1, 'Slams': 1, 'tennis': 1, 'ranking': 1, 'rank': 1, 'gymnastics': 1, 'gymnast': 1, 'aerobics': 1, 'innings': 1, 'runs': 1, 'catch': 1, 'quaterback': 1, 'played': 1, 'lead': 1, 'hits': 1, 'soccer': 1, 'umpire': 1, 'goalkeeper': 1, 'bat': 1, 'racquet': 1, 'ping-pong': 1, 'played': 1, 'pitch': 1, 'field': 1, 'scratch': 1, 'table tennis': 1, 'strikeout': 1,
        'merry': 2, 'christmas': 2, 'halloween': 2, 'mardi gras': 2, 'thanksgiving': 2, 'Coachella': 2, 'Sundance': 2, 'Burning Man': 2, 'Easter': 2, 'family': 2, 'sister': 2, 'couple': 2, 'mother': 2, 'love': 2, 'heart': 2, 'passion': 2, 'brother': 2, 'house': 2, 'household': 2, 'miss': 2, 'buddies': 2, 'friends': 2, 'hopes': 2, 'dreams': 2, 'aspirations': 2, 'fun': 2, 'time': 2, 'mom': 2, 'relationship': 2, 'home': 2, 'soar': 2, 'aim': 2, 'beautiful': 2, 'free': 2,
        'food': 3, 'eat': 3, 'fun': 3, 'chicken': 3, 'hungry': 3, 'starving': 3, 'muffins': 3, 'meat': 3, 'vegetarian': 3, 'non-veg': 3, 'chicken': 3, 'dish': 3, 'salad': 3, 'foodgasm': 3, 'coffee': 3, 'starbucks': 3, 'burgers': 3, 'pizza': 3, 'starved': 3, 'green': 3, 'boil': 3, 'heat': 3, 'restro': 3, 'restaurant': 3, 'gulp': 3, 'health': 3, 'healthy': 3, 'cancer': 3, 'disease': 3, 'migraine': 3, 'headache': 3, 'unwell': 3, 'fine': 3, 'feel': 3, 'nausea': 3, 'body': 3, 'exercise': 3, 'diet': 3, 'puke': 3, 'ill': 3, 'workout': 3, 'fitness': 3, 'cramps': 3, 'woman': 3, 'periods': 3, 'unfit': 3, 'unhealthy': 3, 'healthy': 3, 'eyes': 3,
        'music': 4, 'karoake': 4, 'song': 4, 'jazz': 4, 'blues': 4, 'rock': 4, 'hip hop': 4, 'metal': 4, 'rhythm': 4, 'country': 4, 'soundtrack': 4, 'instrumental': 4, 'progressive': 4, 'acoustic': 4, 'vocal': 4, 'techno': 4, 'romantic': 4, 'rap': 4, 'dub': 4, 'violin': 4, 'guitar': 4, 'instruments': 4, 'listen': 4, 'radio': 4, 'jam': 4, 'jamming': 4, 'spotify': 4, 'apple music': 4, 'pop': 4, 'play': 4, 'perform': 4,
        'writer': 4, 'write': 4, 'novel': 4, 'sing': 4, 'vocal': 4, 'like': 4, 'book': 4, 'draw': 4, 'paint': 4, 'art': 4, 'drum': 4, 'violin': 4, 'guitar': 4, 'guitarist': 4, 'dancer': 4, 'dance': 4, 'climber': 4, 'ballet': 4, 'painter': 4, 'teach': 4, 'teacher': 4, 'photo': 4, 'click': 4, 'photography': 4, 'read': 4, 'create': 4, 'creative': 4,
        'youtube': 5, 'facebook': 5, 'netflix': 5, 'social': 5, 'media': 5, 'videos': 5, 'instagram': 5, 'snapchat': 5, 'chat': 5, 'spotify': 5, 'apple': 5, 'tinder': 5, 'linkedin': 5, 'likes': 5, 'cyber': 5, 'DM': 5, 'ping': 5, 'html': 5, 'www': 5, 'direct messaging': 5,
        'kill': 6, 'dead': 6, 'dying': 6, 'violence': 6, 'guns': 6, 'bully': 6, 'cyber': 6, 'depression': 6, 'sad': 6, 'angry': 6, 'hurt': 6, 'accident': 6, 'racist': 6, 'apologize': 6, 'delay': 6, 'wounded': 6, 'hurtful': 6, 'wound': 6, 'blood': 6, 'needs': 6, 'lies': 6, 'emergency': 6, 'woeful': 6, 'plight': 6, 'sexual': 6, 'hatred': 6, 'assault': 6, 'perpetrators': 6, 'fake': 6, 'Negative': 6, 'vile': 6, 'evil': 6, 'trouble': 6, 'dark': 6,
        'code': 7, 'science': 7, 'tech': 7, 'technology': 7, 'gadgets': 7, 'invent': 7, 'discover': 7, 'program': 7, 'geek': 7, 'earphone': 7, 'headphone': 7, 'specification': 7, 'specs': 7,
        'climate': 7, 'earth': 7, 'school': 7, 'university': 7, 'engineering': 7, 'skill': 7, 'riordan': 7, 'studies': 7, 'study': 7, 'opinions': 7, 'degrees': 7, 'service': 7, 'help': 7, 'assignment': 7, 'supernatural': 7,
        'israel': 8, 'war': 8, 'soldier': 8, 'lebanese': 8, 'greek': 8, 'veteran': 8, 'politics': 8, 'speech': 8, 'leaders': 8, 'issue': 8, 'army': 8, 'military': 8, 'doctrine': 8, 'world': 8, 'Bravo': 8, 'Navy': 8, 'Marine': 8
    }

    '''Create the dictionary of the seed words corpus'''

    dictionary2 = gensim.corpora.Dictionary(
        [['travel', 'mountain', 'place', 'beach', 'hill', 'sun', 'ride', 'car', 'hike', 'hotel', 'arizona', 'tempe', 'water', 'climb', 'journey', 'passerby', 'holiday', 'USA', 'america', 'Everest', 'nature', 'trek', 'walk', 'terrain', 'traveller', 'destination', 'adventure', 'trip', 'itinerary', 'naturelover', 'peaceful', 'beautiful', 'routes', 'restaurant', 'resort', 'picnic', 'greenery', 'plants', 'food', 'energized', 'breathtaking', 'view', 'Mountaineering', 'Hiking', 'Biking', 'Driving', 'trekking', 'diving', 'rome'],
         ['game', 'team', 'win', 'play', 'season', 'second', 'victory', 'sports', 'cycle', 'archer', 'fence', 'goal', 'training', 'ball', 'triathlon', 'basketball', 'baseball', 'football', 'base', 'love', 'passion', 'Slams', 'tennis', 'ranking', 'rank',
             'gymnastics', 'gymnast', 'aerobics', 'innings', 'runs', 'catch', 'quaterback', 'played', 'lead', 'hits', 'soccer', 'umpire', 'goalkeeper', 'bat', 'racquet', 'ping-pong', 'played', 'pitch', 'field', 'scratch', 'table tennis', 'strikeout'],
            ['merry', 'christmas', 'halloween', 'mardi gras', 'thanksgiving', 'Coachella', 'Sundance', 'Burning Man', 'Easter', 'family', 'sister', 'couple', 'mother', 'love', 'heart', 'passion',
                'brother', 'house', 'household', 'miss', 'buddies', 'friends', 'hopes', 'dreams', 'aspirations', 'fun', 'time', 'mom', 'relationship', 'home', 'soar', 'aim', 'beautiful', 'free'],
         ['food', 'eat', 'chicken', 'hunger', 'fun', 'starve', 'muffins', 'meat', 'vegetarian', 'non-veg', 'chicken', 'dish', 'salad', 'foodgasm', 'coffee', 'starbucks', 'burgers', 'pizza', 'starved', 'green', 'boil', 'heat', 'restro', 'restaurant',
             'gulp', 'health', 'healthy', 'cancer', 'disease', 'migraine', 'headache', 'unwell', 'fine', 'feel', 'nausea', 'body', 'exercise', 'diet', 'puke', 'ill', 'workout', 'fitness', 'cramps', 'woman', 'periods', 'unfit', 'unhealthy', 'healthy', 'eyes'],
         ['music', 'karoake', 'song', 'jazz', 'blues', 'rock', 'hip hop', 'metal', 'rhythm', 'country', 'soundtrack', 'instrumental', 'progressive', 'acoustic', 'vocal', 'techno', 'romantic', 'rap', 'dub', 'violin', 'guitar', 'instruments', 'listen', 'radio', 'jam', 'jamming', 'spotify',
             'apple music', 'pop', 'play', 'perform', 'writer', 'write', 'novel', 'sing', 'vocal', 'like', 'book', 'draw', 'paint', 'art', 'drum', 'violin', 'guitar', 'guitarist', 'dancer', 'dance', 'climber', 'ballet', 'painter', 'teach', 'teacher', 'photo', 'click', 'photography'],
            ['youtube', 'facebook', 'netflix', 'social', 'media', 'videos', 'instagram', 'snapchat', 'chat',
                'spotify', 'apple', 'tinder', 'linkedin', 'likes', 'cyber', 'dm', 'ping', 'www', 'direct messaging'],
            ['kill', 'dead', 'dying', 'violence', 'guns', 'bully', 'cyber', 'depression', 'sad', 'angry', 'hurt', 'accident', 'racist', 'apologize', 'delay', 'wounded', 'hurtful',
                'wound', 'blood', 'needs', 'lies', 'emergency', 'woeful', 'plight', 'sexual', 'hatred', 'assault', 'perpetrators', 'fake', 'Negative''vile', 'evil', 'trouble', 'dark'],
            ['code', 'science', 'tech', 'technology', 'gadgets', 'invent', 'discover', 'program', 'geek', 'earphone', 'headphone', 'specification', 'specs', 'climate',
                'earth', 'school', 'university', 'engineering', 'skill', 'riordan', 'studies', 'study', 'opinions', 'degrees', 'service', 'help', 'assignment', 'supernatural'],
            ['israel', 'war', 'soldier', 'lebanese', 'greek', 'veteran', 'politics', 'speech', 'leaders', 'issue', 'army', 'military', 'doctrine', 'world', 'Bravo', 'Navy', 'Marine']])
    # ['god','evidence','believe','reason','faith','exist','bible','religion','claim','spiritual','learnings]]

    '''Call the training and test modules'''
    eta = create_eta(apriori_harder, dictionary2, 9)
    model, df_topics, topic_dict = test_eta(eta, dictionary2, 9, texts, data)

    # remove unwanted characters, numbers and symbols
    df_topics['cleanDocument'] = df_topics['documents'].str.replace(
        "[^a-zA-Z#]", " ")
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')

    # function to remove stopwords
    def remove_stopwords(rev):
        rev_new = " ".join([i for i in rev if i not in stop_words])
        return rev_new

    # remove stopwords from the te xt
    df_topics['cleanDocument'] = [remove_stopwords(
        r.split()) for r in df_topics['cleanDocument']]

    #  iterate over for each topic:(finding the main categorycd b
    # )
    category_dict = {'0': [], '1': [], '2': [], '3': [],
                     '4': [], '5': [], '6': [], '7': [], '8': []}
    for index, row in df_topics.iterrows():
        # print(row['documents'])
        lis = row['documents'].split(' ')
        lis = [(ind.lower()).capitalize() for ind in lis]
        row['documents'] = ' '.join(lis)
        if row['documents'] != 'nan':
            category_dict[str(row['topics'])].append(row['documents'])

    nlp = spacy.load('en_core_web_lg')
    main_dict, count = {}, 0
    for id, value in category_dict.items():
        subcategory_dict = {'events0': [], 'entities0': [], 'others0': [], 'location0': [],
                            'events1': [], 'entities1': [], 'others1': [], 'location1': [],
                            'events2': [], 'entities2': [], 'others2': [], 'location2': [],
                            'events3': [], 'entities3': [], 'others3': [], 'location3': [],
                            'events4': [], 'entities4': [], 'others4': [], 'location4': [],
                            'events5': [], 'entities5': [], 'others5': [], 'location5': [],
                            'events6': [], 'entities6': [], 'others6': [], 'location6': [],
                            'events7': [], 'entities7': [], 'others7': [], 'location7': [],
                            'events8': [], 'entities8': [], 'others8': [], 'location8': [],
                            'events9': [], 'entities9': [], 'others9': [], 'location9': []}
        for doc in value:
            string1 = nlp(doc)
            for ent in string1.ents:
                subcategory_dict['others'+str(count)] = topic_dict[str(id)]
                # print(ent.text, ent.start_char, ent.end_char, ent.label_)

                # Categorise irrelevant words:
                text = ent.text
                string_notwords, string_words = '', ''
                main_string = []
                for word in text.split(' '):
                    word = word.replace('#', '')
                    # single characters
                    # word = re.sub(r"(\w)\1*", r'\1', word)
                    # Handle double or more characters
                    word = re.sub(r'(.)\1{3,}', r'\1', word)
                    main_string.append(word.lower())
                    if word in words.words() and word.isdigit() == False:
                        string_words += word + ' '
                    elif word not in words.words() and word.isdigit() == False and string_words != '':
                        string_notwords += word + ' '
                text = ' '.join(main_string)
                if ent.label_ == 'EVENT' or ent.label_ == 'NORP' or ent.label_ == 'ORG':
                    if len(subcategory_dict['events' + str(count)]) <= 20:
                        if len(string_words) > 2 and len(text.split(' ')) > 2 and len(text.split(' ')) <= 5:
                            subcategory_dict['events'+str(count)].append(text)
                elif ent.label_ == 'PERSON':
                    if len(subcategory_dict['entities'+str(count)]) <= 20:
                        if string_notwords != '' and len(string_notwords.split(' ')) == 1 and string_notwords not in subcategory_dict['entities'+str(count)]:
                            subcategory_dict['entities' +
                                             str(count)].append(string_notwords)
                        if len(text.split(' ')) == 2 and string_notwords != '' and text not in subcategory_dict['entities'+str(count)]:
                            subcategory_dict['entities' +
                                             str(count)].append(text)
                elif ent.label_ == 'GPE' or ent.label_ == 'LOC' or ent.label_ == 'FAC':
                    if len(subcategory_dict['location'+str(count)]) <= 20:
                        # if len(text.split()) != 0 and len(text.split(' ')) <=5:
                        subcategory_dict['location'+str(count)].append(text)
                else:
                    pass

        count += 1
        if id == '0':
            main_dict['Travelling'] = subcategory_dict
        elif id == '1':
            main_dict['Sports'] = subcategory_dict
        elif id == '2':
            main_dict['Life'] = subcategory_dict
        elif id == '3':
            main_dict['Health'] = subcategory_dict
        elif id == '4':
            main_dict['Recreation'] = subcategory_dict
        elif id == '5':
            main_dict['SocialMedia'] = subcategory_dict
        elif id == '6':
            main_dict['Naysays'] = subcategory_dict
        elif id == '7':
            main_dict['General'] = subcategory_dict
        elif id == '8':
            main_dict['Politics'] = subcategory_dict

    '''Doing it in proper format:Categories and subcategories:'''

    import random
    category_list = ["Travelling", "Health", "General", "Naysays",
                     "Sports", "Recreation", "SocialMedia", "Life", "Politics"]

    ids = []
    parents = []
    user = ""
    sun_len, count_cat = 0, 0
    keys = list(main_dict.keys())
    # random.shuffle(keys)
    for key in keys:
        if key in category_list and main_dict[key] != {}:
            count_cat += 1
            count = 0
            if count_cat <= 7:
                ids.append(key)
                parents.append(user)
                for k, v in main_dict[key].items():
                    if v:
                        count += 1
                        ids.append(k)
                        parents.append(key)
                        random.shuffle(v)
                        for item in v[:20]:
                            ids.append(item.strip())
                            parents.append(k)
        # if all lists with values, that grandparent were empty, just delete the appended grandparent
        if count == 0:
            del ids[-1]
            del parents[-1]

        if sun_len <= 7:
            sun_len += 1
        else:
            break

    '''naming the labels'''
    labels = []
    for index in range(len(ids)):
        id = ids[index]
        if id in ['events0', 'events1', 'events2', 'events3', 'events4', 'events5', 'events6', 'events7', 'events8', 'events9']:
            labels.append('Communities')
        elif id in ['entities0', 'entities1', 'entities2', 'entities3', 'entities4', 'entities5', 'entities6', 'entities7', 'entities8', 'entities9']:
            labels.append('People')
        elif id in ['location0', 'location1', 'location2', 'location3', 'location4', 'location5', 'location6', 'location7', 'location8', 'location9']:
            labels.append('Locations')
        elif id in ['others0', 'others1', 'others2', 'others3', 'others4', 'others5', 'others6', 'others7', 'others8']:
            labels.append('Others')
        else:
            labels.append(id)

    '''return the results for the chat'''
    interest_chart = {'ids': [], 'parents': [], 'labels': []}
    interest_chart['ids'] = ids
    interest_chart['parents'] = parents
    interest_chart['labels'] = labels
    return interest_chart
