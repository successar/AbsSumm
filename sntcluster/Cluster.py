#!/usr/bin/python
"""
sentence clustering
"""

import os
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from tokenx import tokenize, fetch_sents
import sim
import numpy as np

class SegSnt:
    def __init__(self, seg_txt, word_freq):
        self.seg_txt = seg_txt
        self.word_freq = word_freq
        self.vector = None

    def set_vector(self, vector):
        self.vector = vector

class SntDoc:
    def __init__(self, text, seg_snts):
        self.text = text
        self.seg_snts = seg_snts
        self.vector = None

    def set_snts(self, seg_snts):
        self.seg_snts = seg_snts

    def set_vector(self) :
        self.vector = [0]
        for snt in self.seg_snts:
            self.vector += snt.vector

# preprocess the document set
def preproc(docs):
    snt_docs = []
    for doc in docs:
        seg_snts = []
        snts = fetch_sents(doc)
        for snt in snts:
            if len(snt) > 10:
                word_list = tokenize(snt)
                seg_txt = snt
                word_freq = stat_word_freq(word_list)
                seg_snts.append(SegSnt(seg_txt, word_freq))
        snt_docs.append(SntDoc(doc, seg_snts))
    return snt_docs

def stat_word_freq(words):
    w_freq = {}
    for word in words:
        if not w_freq.has_key(word): w_freq[word] = 0
        w_freq[word] += 1
    return w_freq

def main(docs):
    snt_docs = preproc(docs)
    w_freqs = []
    for snt_doc in snt_docs:
        for seg_snt in snt_doc.seg_snts:
            w_freqs.append(seg_snt.word_freq)

    dvec = DictVectorizer(sparse=False)
    X = dvec.fit_transform(w_freqs)
    #print X

    tfvec = TfidfTransformer()
    tfvec.fit_transform(X)

    i = 0
    for snt_doc in snt_docs:
        for seg_snt in snt_doc.seg_snts:
            vector = X[i, :]
            i += 1
            seg_snt.set_vector(vector)
        snt_doc.set_vector()

    imp_doc, m_index = find_imp_doc(snt_docs)
    means = [] #initial means

    #print ("--------------- Most Important Document -----------------------" + str(m_index))
    for seg_snt in imp_doc.seg_snts:
        #print seg_snt.seg_txt
        if not ndarray_in(seg_snt.vector, means):
            means.append(seg_snt.vector)
    
    snt_clusters = kmeans(snt_docs, means=means)

    return snt_clusters

# find the most important document
def find_imp_doc(snt_docs):
    max_score = -10000.0
    m_index = -1
    for i in range(len(snt_docs)):
        score = 0.0
        for j in range(len(snt_docs)):
            if(i != j) :
                sdi, sdj = snt_docs[i], snt_docs[j]
                score += calc_doc_sim(sdi, sdj)
        if score > max_score:
            max_score = score
            m_index = i
    return snt_docs[m_index],m_index

# calculate the similarity between two documents
def calc_doc_sim(sdoci, sdocj):    
    return sim.cos(sdoci.vector, sdocj.vector)

def ndarray_in(one, ndarrays):
    for ndarray in ndarrays:
        if (one==ndarray).all(): return True
    return False

def kmeans(snt_docs, means):
    clusters = [[] for mean in means]
    for snt_doc in snt_docs:
        for seg_snt in snt_doc.seg_snts:
            index , max_score = find_cluster(seg_snt, means)
            if index > -1 and max_score >= 0.4 :
                clusters[index].append(seg_snt)

    return [cluster for cluster in clusters if len(cluster) >= (len(snt_docs))/3.0]

# find a cluster that it belongs to
def find_cluster(seg_snt, means):
    max_score = -10000.0
    m_incex = -1
    for i in range(len(means)):
        score = sim.cos(seg_snt.vector, means[i])
        if score > max_score:
            max_score = score
            m_incex = i
    return m_incex,max_score