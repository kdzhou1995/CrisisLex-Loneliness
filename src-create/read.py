# CrisisLex
# Author: Alexandra Olteanu
# Check LICENSE for details about copyright.

import csv
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

def get_stemmed_terms_list(doc, stem_words_map = None, stem_bigrams_map = None):
    ps = PorterStemmer()
    local_map = dict()
    word_list = []

    # split tweet into word list
    clean_doc = [(w.strip()).lower() for w in doc.split() if len(w) in range(3,16)]

    # filters stop words
    filtered_words = [w.strip('.,;?!:)(#') for w in clean_doc if not w.strip('.,;?!:)(#') in stopwords.words('english')]

    # stem words
    for w in filtered_words:
        if w.isalpha():
            w_temp = ps.stem(w)
            if stem_words_map is not None:
                if w_temp not in stem_words_map:
                    stem_words_map[w_temp] = dict()
                stem_words_map[w_temp][w] = stem_words_map[w_temp].get(w, 0)+1
                local_map[w_temp] = w
            word_list.append(w_temp)

    # creat bigrams
    bigrams = nltk.bigrams(word_list)
    for b in bigrams:
        bigram_org = (local_map[b[0]],local_map[b[1]])
        if stem_bigrams_map is not None:
                if b not in stem_bigrams_map:
                    stem_bigrams_map[b] = dict()
                stem_bigrams_map[b][bigram_org] = stem_bigrams_map[b].get(bigram_org, 0)+1

    return word_list, bigrams

# keeps track of the exact form of the stemmed bigrams, not only the one of the words
def get_tweet_terms(tweet, stem_map = None, bigrams_map = None):
    
    # stem raw tweet and put the terms into list
    words, bigrams = get_stemmed_terms_list(tweet, stem_map, bigrams_map)
    filtered_words = [w for w in words if not w in stopwords.words('english')]

    # replaces bigrams by using nltk to turn the filtered word list into a list of bigrams
    bigrams = nltk.bigrams(filtered_words) # what was the purpose of getting bigrams in this method then if we aren't doing anything with bigrams map?
    words_set = set(filtered_words)
    terms_dict = {}

    # combines unigrams and bigrams into terms dict with the y value to indicate they appeared in a tweet
    for w in words_set:
        terms_dict['%s'%w] = 'y'

    for b in bigrams:
        terms_dict['%s %s'%(b[0],b[1])] = 'y'

    return terms_dict

def get_terms(ifile, stem_map = None, bigrams_map = None, min_occurence = 0.001):
    tweets_cls = [] # class
    tweets_type = [] #
    tweets_terms = [] # list of tweets that have been broken down to stemmed unigram and bigrams
    tweets_id = []
    tweets_no = 0
    ws = set() # a set of words that have occurred over the entirety of the corpus
    wd_occ = dict() ## term (unigram and bigram) co-occurence in the same tweet
    fd = nltk.FreqDist() # lists the counts of each term (unigram bigram) over the entire corpus
    r = csv.reader(ifile)

    print("Reading...")
    headers = next(r)
    for tokens in r:
        tweets_no += 1
        id = tokens[0].strip()  #debug note: tweet id    
        tweet = tokens[1].strip()   #debug note: tweet text
        cls = tokens[2].strip()     #debug note: class
        terms = get_tweet_terms(tweet, stem_map, bigrams_map) # extracts bigrams and unigrams as well as appearence indicator 'y' from raw tweet 

        tweets_cls.append(cls) # add class
        tweets_type.append(type) # add type?
        tweets_id.append((id,tweet)) # add tweet id
        tweets_terms.append(terms) # add the unigram/bigram - 'y' dictionary to term list
        for t in terms: # for each term, update the count in the frequency distribution
            fd[t] += 1
        ws.update(set(terms.keys())) # add each term into word set
    print(f"... {tweets_no} tweets")

    print("Cleaning...")
    for t in tweets_terms:
        s = set()
        term = list(t.keys()) # gets unigrams and bigrams from the terms dict of a single tweet
        l = len(term) # total length of the list of tweet's unigrams and bigrams
        for i,f in enumerate(term):
            if fd[f] <= min_occurence*tweets_no: # check in the frequency distribution to see if the frequency (count) is less than the minimum occurence filter. min occurence scales with corpus size (document numbers)
                s.add(f) # filters and only adds to set s terms that occur blow the minumum allowed frequency for removal later
            if l > 1: # if total length of the tweet is greater than 1
                if i == l-1: # if last term, exit loop
                    break
                for j in range(i+1, l):
                    # this seems to be updating word co-occurence
                    #(current word, next word) freq += 1, 
                    # current word stays the same but 
                    # every combination with another word in the tweet gets keyed into the dictionary and count gets updated 
                    wd_occ[(f, term[j])] = wd_occ.get((f, term[j]), 0)+1 
        
        # remove the terms with too low frequency from tweet terms, and from word set
        for f in s:
            del t[f]
            ws.discard(f)

    return tweets_cls, tweets_terms, wd_occ, ws, fd