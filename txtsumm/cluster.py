from sntcluster import Cluster
from nltk.tokenize import sent_tokenize
from Summ import generateSummaries
import os
import re

PROJECT_DIR=os.path.dirname(os.path.abspath(__file__))+"/../"
RESOURCES_DIR=PROJECT_DIR+"resources/"


import kenlm
lm = kenlm.Model(RESOURCES_DIR+'/lm-3g.km')

if __name__ == "__main__": 
	in_dir = PROJECT_DIR + "txtsumm/input"
	out_dir = PROJECT_DIR + 'txtsumm/output2'
	dirs = os.listdir(in_dir)
	for d in dirs :
		d_n = in_dir + '/' + d
		files = os.listdir(d_n)
		docs = []
		for f in files:
			fp = open(d_n + '/' + f, "r")
			doc = fp.read().decode('utf-8')
			#doc = doc.partition('<TEXT>')[2].partition('</TEXT>')[0]
			doc = doc.replace('\n', ' ')
			docs.append(doc)

		print d, len(docs)
	    
		clusters = Cluster.main(docs) # cluster op
		# index = 0
		# for cluster in clusters:
		#     print('-------cluster %d-------\n' % index)
		#     for seg_snt in cluster:
		#         print(seg_snt.seg_txt.encode('utf-8') + '\n')
		#     index += 1

		sent_clusters = []
		for cluster in clusters :
			sent = []
			for seg_snt in cluster:
				sent.append(seg_snt.seg_txt.encode('ascii', 'ignore'))
			sent_clusters.append(sent)

		print "Clustering Done ... " + str(len(clusters))

		summ = generateSummaries(sent_clusters, lm = lm, filename = d, mode="Abstractive")

		f = open(out_dir + '/' + d + '_output.txt', 'w')
		f.write(summ)
		f.close()

		print '\n'

