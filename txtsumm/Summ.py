# -*- coding: utf-8 -*-

import absummarizer.takahe as tk
import os
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

PROJECT_DIR=os.path.dirname(os.path.abspath(__file__))+"/../"
print "Project dir", PROJECT_DIR
RESOURCES_DIR=PROJECT_DIR+"resources/"

def sentenceCapitalize(sent):
    sentences = sent.split(". ")
    sentences2 = [sentence[0].capitalize() + sentence[1:] for sentence in sentences]
    string2 = '. '.join(sentences2)
    return string2
    
def txtFromSents(finalsummarySents):
    txtSummary=""
    if finalsummarySents is None:
        return txtSummary
    for sent in finalsummarySents:
        if sent.strip().endswith("."):
            sent=sentenceCapitalize(sent)
        else:
            sent=sentenceCapitalize(sent)+"."
        sent=sent.replace(":.",".")
        txtSummary=txtSummary+"\n"+sent
        txtSummary=txtSummary.strip()
    return txtSummary


def generateSummaries(clusters, lm, filename, mode = "Extractive"):
    '''
    This is where the ILP works to select the best sentences and form the summary
    '''
    if mode == "Abstractive":
        '''
        Here sentences should have POS tagged format
        '''
        genSentences_total = []
        for n_c, sentences in enumerate(clusters):
            taggedsentences=[]
            for sent in sentences: 
                sent=sent.decode('utf-8','ignore')
                tagged_sent=''
                tagged_tokens = pos_tag(word_tokenize(sent))
                for token in tagged_tokens:
                    word, pos=token
                    tagged_sent=tagged_sent+' '+word+"/"+pos
                taggedsentences.append(tagged_sent.strip())
                
            genSentences_total.append(tk.getSentences(taggedsentences, cluster_name=filename+"_"+str(n_c)))

        print "Word Graphs Done ... "
        finalSentencesRetained=tk.solveILPFactBased(genSentences_total,lm, mode="Abstractive")    
        
        summary=txtFromSents(finalSentencesRetained)
        print "=======Summary:===== \n", summary.encode('utf-8', errors='ignore')

    if mode == "Extractive":
        lm=[] 
        genSentences_total=[]
        for sentences in clusters :
            taggedsentences=[]
            for sent in sentences: 
                sent=sent.decode('utf-8','ignore')
                taggedsentences.append(sent.strip())
                
            genSentences_total.append(tk.getSentences(taggedsentences, mode="Extractive"))

        finalSentencesRetained=tk.solveILPFactBased(genSentences_total, lm, mode="Extractive")
    
        summary=txtFromSents(finalSentencesRetained)
        print "=======Summary:===== \n", summary.encode('utf-8')

    return summary