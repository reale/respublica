# -*- coding: utf-8 -*-
from sklearn.decomposition import KernelPCA
from sklearn.metrics.pairwise import pairwise_distances
from scipy.stats.mstats import zscore
import glob
import json
import re
import datetime
import os
import cPickle
import codecs
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import double,zeros

def get_news(sources=['corriere', 'fatto', 'giornale', 'libero', 'repubblica', 'sole24ore']):
    '''
    Collects all news articles from political ressort of six major Italian newspapers
    Articles are transformed to BoW vectors and assigned to a political party
    For better visualization, articles' BoW vectors are also clustered into topics

    INPUT
    folder      the model folder containing classifier and BoW transformer
    sources     a list of strings for each newspaper for which a crawl is implemented

    '''
    import classifier
    from bs4 import BeautifulSoup
    from api import fetch_url, basic_fetch_url
    import urllib2
    
    articles = []
    
    # the classifier for prediction of political attributes 
    clf = classifier.Classifier(train=False)
    
    for source in sources:

        if source is 'corriere':
            # fetching articles from 'Corriere della Sera'
            url = 'https://www.corriere.it/politica/'
            site = BeautifulSoup(basic_fetch_url(url))
            titles = site.findAll("h3", { "class" : "title_art" })
            urls = ['https:'+a.findNext('a')['href'] for a in titles]
         
        if source is 'fatto':
            # fetching articles from 'Il Fatto Quotidiano'
            url = 'https://www.ilfattoquotidiano.it/politica-palazzo/'
            site = BeautifulSoup(basic_fetch_url(url))
            titles = site.findAll("h2", { "class" : "title" })
            urls = [a.findNext('a')['href'] for a in titles]
         
        if source is 'giornale':
            # fetching articles from 'il Giornale'
            url = 'http://www.ilgiornale.it/sezioni/interni.html'
            site = BeautifulSoup(basic_fetch_url(url))
            titles = site.findAll("h2", { "class" : "entry-title" })
            urls = ['http://www.ilgiornale.it'+a.findNext('a')['href'] for a in titles]

        if source is 'libero':
            # fetching articles from 'Libero'
            url = 'http://www.liberoquotidiano.it/sezioni/14/politica'
            site = BeautifulSoup(basic_fetch_url(url))
            titles = site.findAll("h2", { "class" : "titolo" })
            urls = [a.findNext('a')['href'] for a in titles]

        if source is 'repubblica':
            # fetching articles from 'la Repubblica'
            url = 'https://www.repubblica.it/politica/'
            site = BeautifulSoup(basic_fetch_url(url))
            titles = site.findAll("h2", { "class" : "entry-title" })
            urls = [a.findNext('a')['href'] for a in titles]
         
        if source is 'sole24ore':
            # fetching articles from 'Il Sole 24 Ore'
            url = 'http://www.ilsole24ore.com/notizie/politica.shtml'
            site = BeautifulSoup(basic_fetch_url(url))
            titles = site.findAll("h3", { "class" : "mid" })
            urls = ['http://www.ilsole24ore.com'+a.findNext('a')['href'] for a in titles]

        print "Found %d articles on %s"%(len(urls),url)
         
        # predict party from url for this source
        print "Predicting %s"%source
        for url in urls:
            try:
                title,text = fetch_url(url)
                prediction = clf.predict(text)
                prediction['url'] = url
                prediction['source'] = source
                articles.append((title,prediction))
            except:
                print('Could not get text from %s'%url)
                pass

    # do some topic modeling
    topics = kpca_cluster(map(lambda x: x[1]['text'][0], articles))
  
    # remove original article text for faster web-frontend
    for a in articles:
        a[1]['text'] = 'deleted'

    # store current news and topics
    json.dump(articles,open('news.json','wb'))
    json.dump(topics,open('topics.json','wb'))

def load_sentiment(negative='SentiWS_v1.8c/SentiWS_v1.8c_Negative.txt',\
        positive='SentiWS_v1.8c/SentiWS_v1.8c_Positive.txt'):
    words = dict()
    for line in open(negative).readlines():
        parts = line.strip('\n').split('\t')
        words[parts[0].split('|')[0]] = double(parts[1])
        if len(parts)>2:
            for inflection in parts[2].strip('\n').split(','):
                words[inflection] = double(parts[1])
    
    for line in open(positive).readlines():
        parts = line.strip('\n').split('\t')
        words[parts[0].split('|')[0]] = double(parts[1])
        if len(parts)>2:
            for inflection in parts[2].strip('\n').split(','):
                words[inflection] = double(parts[1])
   
    return words

def get_sentiments(data):
    
    # filtering out some noise words
    stops = map(lambda x:x.lower().strip(),open('stopwords.txt').readlines()[6:])

    # vectorize non-stopwords 
    bow = TfidfVectorizer(min_df=2,stop_words=stops)
    X = bow.fit_transform(data)

    # map sentiment vector to bow space
    words = load_sentiment()
    sentiment_vec = zeros(X.shape[1])
    for key in words.keys():
        if bow.vocabulary_.has_key(key):
            sentiment_vec[bow.vocabulary_[key]] = words[key]
    
    # compute sentiments 
    return X.dot(sentiment_vec)


def kpca_cluster(data,nclusters=20,topwhat=10):
    '''

    Computes clustering of bag-of-words vectors of articles

    INPUT
    folder      model folder
    nclusters   number of clusters

    '''
    from sklearn.cluster import KMeans
    # filtering out some noise words
    stops = map(lambda x:x.lower().strip(),codecs.open('data/stopwords.txt',"r","utf-8").readlines()[6:])

    # vectorize non-stopwords 
    bow = TfidfVectorizer(min_df=4,stop_words=stops)
    X = bow.fit_transform(data)

    # creating bow-index-to-word map
    idx2word = dict(zip(bow.vocabulary_.values(),bow.vocabulary_.keys()))
   
    # compute clusters
    km = KMeans(n_clusters=nclusters).fit(X)

    clusters = []
    for icluster in range(nclusters):
        nmembers = (km.labels_==icluster).sum()
        if nmembers > 1: # only group clusters big enough but not too big
            members = (km.labels_==icluster).nonzero()[0]
            topwordidx = km.cluster_centers_[icluster,:].argsort()[-topwhat:][::-1]
            topwords = ' '.join([idx2word[wi] for wi in topwordidx])
            #print u'Cluster %d'%icluster + u' %d members'%nmembers + u'\n\t'+topwords
            clusters.append({
                'name':'Cluster-%d'%icluster,
                'description': topwords,
                'members': list(members),
                })

    return clusters

def write_distances_json(folder='model'):
    articles, data = get_news()
    distances_json = {
            'articles': articles,
            'distances': [
                { 'name': dist, 'distances': pairwise_dists(data) } for dist in dists
            ],
            'clusterings': [
                { 'name': 'Parteivorhersage', 'clusters': party_cluster(articles) },
                { 'name': 'Ähnlichkeit', 'clusters': kpca_cluster(data,nclusters=len(articles)/2,ncomponents=40,zscored=False) },
            ]
        }

    # save article with party prediction and distances to closest articles
    datestr = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    open(folder+'/distances-%s'%(datestr)+'.json', 'wb').write(json.dumps(distances_json))
    # also save that latest version for the visualization
    open(folder+'/distances.json', 'wb').write(json.dumps(distances_json))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(\
        description='Downloads, transforms and clusters news articles')

    parser.add_argument('-p','--distances',help='If pairwise distances of text should be computed',\
            action='store_true', default=True)
    
    args = vars(parser.parse_args())
    if args['distances']:
        write_distances_json(folder=args['folder'])
