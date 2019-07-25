import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import math

data = pd.read_csv('keyword.csv')['indikasi']
data.index += 1
# data

tfidf = TfidfVectorizer()
wordSet = []
for each in data:
    for ieach in each.split():
        if ieach not in wordSet:
            wordSet.append(ieach)
wordSet = set(wordSet)

data_new = []
wordDict = [None] * len(data)
for row in enumerate(data):
    wordDict[row[0]] = dict.fromkeys(wordSet, 0)
    for word in row[1].split():
        wordDict[row[0]][word]+=1
    data_new.append(wordDict[row[0]])

# pd.DataFrame(data_new)

def computeTF(wordDict, bow):
    tfDict = {}
    bowCount = len(bow)
    for word, count in wordDict.items():
        tfDict[word] = count/float(bowCount)
    return tfDict

tfData = [None] * len(data)
for row in enumerate(data):
    tfData[row[0]] = computeTF(wordDict[row[0]], row[1])

def computeIDF(docList):
    import math
    idfDict = {}
    N = len(docList)
    
    idfDict = dict.fromkeys(docList[0].keys(), 0)
    for doc in docList:
        for word, val in doc.items():
            if val > 0:
                idfDict[word] += 1
    
    for word, val in idfDict.items():
        idfDict[word] = math.log10(N / float(val))
        
    return idfDict

idfs = computeIDF(data_new)

def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val*idfs[word]
    return tfidf

tfidfBow = [None] * len(data)
bow_new = []
for each in enumerate(data):
    tfidfBow[each[0]] = computeTFIDF(tfData[each[0]], idfs)
    bow_new.append(tfidfBow[each[0]])
indexed = pd.DataFrame(bow_new)
# indexed

searching = "lesu dan tidak bergairah"
searching = [s.lower() for s in searching.split()]
index = indexed[searching].sum(axis = 1, skipna = True)
index = pd.DataFrame(index.sort_values(ascending=False), columns=['index rate'])
index.index += 1

allData = pd.DataFrame(data)
final_table = index.join(allData)
indikasi = final_table[final_table['index rate'] > 0.0].index.values.tolist()
indikasi = [str(i) for i in indikasi]
indikasi
# final_table

# Inner Product
metadata1 = pd.read_csv('metadata1.csv').set_index('md1')
result1 = metadata1[indikasi].sum(axis = 1, skipna = True)

metadata2 = pd.read_csv('metadata2.csv').set_index('md2')
value1 = result1.values.tolist()

innerProductresult = pd.DataFrame((metadata2 * value1).sum(axis = 1, skipna = True), columns={'value'})
# innerProductResult = innerProductresult.join(data)[['indikasi','value']]

highestInnerProductRate = (metadata2 * value1).sum(axis = 1, skipna = True).idxmax()
data[highestInnerProductRate]
# innerProductResult
# value1

# Cosine Similarity
# indikasi = ['1','2','4','5','6','8','9','11']
metadata1 = pd.read_csv('metadata1.csv').set_index('md1')
xy = metadata1[indikasi].sum(axis = 1, skipna = True)
y2 = metadata1.sum(axis = 1, skipna = True)**(1/2)
x2 = len(indikasi)**(1/2)
result1 = xy/(x2*y2)
result1

metadata2 = pd.read_csv('metadata2.csv').set_index('md2')

x22 = math.sqrt((result1**2).sum())
y22 = metadata2.sum(axis = 1, skipna = True)**(1/2)
result1 = result1.values.tolist()
xy2 = (metadata2 * result1).sum(axis = 1, skipna = True)
# xy2 = pd.DataFrame((metadata2 * result1).sum(axis = 1, skipna = True), columns={'value'})
pd.DataFrame(xy2/(x22*y22), columns={'value'}).join(data)[['indikasi','value']]