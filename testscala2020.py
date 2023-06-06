import numpy as np
import pylab as plt
import seaborn as sns; sns.set()
import pickle

import sparseRRR

def preprocess(data):
    X = data['counts'][:,data['mostVariableGenes']] / np.sum(data['counts'], axis=1) * 1e+6
    X = np.array(X)
    X = np.log2(X + 1)
    X = X - np.mean(X, axis=0)
    X = X / np.std(X, axis=0)

    Y = data['ephys']
    Y = Y - np.mean(Y, axis=0)
    Y = Y / np.std(Y, axis=0)
    
    return (X,Y)

data = pickle.load(open('/home/fank/Documents/patch-seq-rrr/data/scala2020.pickle', 'rb'))

X,Y = preprocess(data)
genes = data['genes'][data['mostVariableGenes']]

print('Shape of X:', X.shape, '\nShape of Y:', Y.shape)

w,v = sparseRRR.relaxed_elastic_rrr(X, Y, rank=2, lambdau=.4, alpha=1)

print('\nGenes selected: {}'.format(np.sum(w[:,0]!=0)))
print(', '.join(genes[w[:,0]!=0]))

sparseRRR.bibiplot(X, Y, w, v, titles = ['RNA expression', 'Electrophysiology'],
                   cellTypes=data['ttype'], 
                   cellTypeColors=data['colors'],
                   YdimsNames=data['ephysNames'],
                   XdimsNames=genes)