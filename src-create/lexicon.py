# CrisisLex
# Author: Alexandra Olteanu
# Check LICENSE for details about copyright.

import nltk
import math

class Lexicon:
    def __init__(self, documents, terms, classes, class_types, frequency, main_class, min_docs):
        self.terms = terms  # the terms used to build the lexicon
        self.documents = documents # a corpus, list of tweets whose words have been broken down to stemmed unigram and bigrams
        self.classes = classes # classes of tweets
        self.terms_frequency = frequency # frequency distribution
        self.terms_frequency_per_class = dict() # distribution of words per class
        self.main_class = main_class # the main predicting class, in this case (+) / (lonely)
        # the minimum support for a term (i.e., number of documents in the class of interest in order to be considered)
        self.min_docs = min_docs # minimum document threshold (20)
        self.class_occ = dict() # the count of how many (+) classifications in corpus vs (-) classes
        for c in class_types:
            self.terms_frequency_per_class[c]=nltk.FreqDist() # each class gets a freq distribution
            self.class_occ[c] = classes.count(c) # the count of how many (+), (-) or however many classification groups there are in corpus
        for i, doc in enumerate(self.documents):
            cls = self.classes[i] # gets the class label for this tweet
            for t in doc: # for term in tweet
                self.terms_frequency_per_class[cls][t] += 1 # in dict, get class, get term in class, increment

    # the scoring functions return the list of discriminative terms for the class of interest according to each metric
    def pmi_polarity_metric(self, thr = None):
        terms = {}
        for t in self.terms:
            fr = self.terms_frequency_per_class[self.main_class][t]
            if fr<= self.min_docs:
                continue
            try:
                # tweets that contain t and are in the class
                # relevant, present
                n11 = self.terms_frequency_per_class[self.main_class][t]
                # tweets that contain t and are not in the class; we add 1 to ensure that pmi never defaults to inf
                # relevant, not present
                n01 = self.terms_frequency[t] - n11
                if n11 == 0:
                    pmi = 0
                else:
                    if n01 == 0:
                        pmi = 100
                    else:
                        pmi = math.log((float(n11)/self.class_occ[self.main_class])/(float(n01)/(len(self.documents)-self.class_occ[self.main_class])))
                        if pmi<0:
                            pmi = 0
            except:
                print("I can't compute the crisis score. Do you have enough training data?")
            if thr is None:
                terms[t] = pmi
            else:
                if pmi>=thr:
                    terms[t] = pmi
        return terms

    def chi2_metric(self, thr = None):
        terms = {}
        n = len(self.documents)
        for t in self.terms:
            fr = self.terms_frequency_per_class[self.main_class][t]
            if fr<= self.min_docs:
                continue
            try:
                n11 = self.terms_frequency_per_class[self.main_class][t] # tweets that contain t and are in the class
                n01 = self.terms_frequency[t] - n11 # tweets that contain t and are not in the class
                n10 = self.class_occ[self.main_class] - n11 # tweets that do not contain t and are in the class
                n00 = (n - self.class_occ[self.main_class]) - n01 # tweets that do not contain t and are not in the class
                p_t_pos = float(n11)/self.class_occ[self.main_class]
                p_t_neg = float(n01)/(len(self.documents)-self.class_occ[self.main_class])

                try:
                    chi2 = (n*(n11*n00-n10*n01)*(n11*n00-n10*n01))/((n11+n01)*(n11+n10)*(n10+n00)*(n01+n00))
                except:
                    chi2 = 0
                if p_t_pos<p_t_neg:
                    chi2 = -chi2

                if thr is None:
                    terms[t] = chi2
                else:
                    if chi2>=thr:
                        terms[t] = chi2
            except:
                print("I can't compute the crisis score. Do you have enough training data?")
        return terms

    def frequency_metric(self, thr = None):
        terms = {}
        for t in self.terms:
            fr = self.terms_frequency_per_class[self.main_class][t]
            if fr >=self.min_docs:
                try:
                    p = float(fr)/self.class_occ[self.main_class]
                except:
                    print("I can't compute the crisis score. Do you have enough training data?")

                if thr is None:
                    terms[t] = p
                else:
                    if p>=thr:
                        terms[t] = p
        return terms

    def rsv_metric(self):
        # Robertson's Selection Value (RSV)
        terms = {}
        n = len(self.documents)
        for t in self.terms:
            fr = self.terms_frequency_per_class[self.main_class][t]
            if fr<= self.min_docs:
                continue
            try:
                n11 = self.terms_frequency_per_class[self.main_class][t] # (A) tweets that contain t and are in the class
                n01 = self.terms_frequency[t] - n11 # (B) tweets that contain t and are not in the class
                n10 = self.class_occ[self.main_class] - n11 # (C) tweets that do not contain t and are in the class
                n00 = (n - self.class_occ[self.main_class]) - n01 # (D) tweets that do not contain t and are not in the class

                ## prevent divide by zero errors
                if n01 == 0:
                    n01 = 1
                if n10 == 0:
                    n10 = 1
                if n00 == 0:
                    n00 = 1

                ## calculate RSV
                rsv = n11 * math.log((n11 * n00)/(n01 * n10))

                terms[t] = rsv
            except:
                print("I can't compute the crisis score. Do you have enough training data?")
        return terms
    
    def drc_metric(self):
        # Document and Relevance Correlation (DRC)
        terms = {}
        n = len(self.documents)
        for t in self.terms:
            fr = self.terms_frequency_per_class[self.main_class][t]
            if fr<= self.min_docs:
                continue
            try:
                n11 = self.terms_frequency_per_class[self.main_class][t] # (A) tweets that contain t and are in the class
                n01 = self.terms_frequency[t] - n11 # (B) tweets that contain t and are not in the class
                n10 = self.class_occ[self.main_class] - n11 # (C) tweets that do not contain t and are in the class
                n00 = (n - self.class_occ[self.main_class]) - n01 # (D) tweets that do not contain t and are not in the class

                ## calculate DRC
                ## drc = (A^2)/sqrt(A + B)
                drc = (n11 ** 2) / math.sqrt(n11 + n01)

                terms[t] = drc
            except:
                print("I can't compute the crisis score. Do you have enough training data?")
        return terms