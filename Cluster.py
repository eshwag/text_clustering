import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
import time

class text_Clustering(object):
    ''' clusters text with different techniques.'''
    
    def __init__(self):
        pass
    
    class lsa(object):
        ''' latent semanic analysis
        Parameters:
        ----------
        data: tfidf matrix (term frequency inverse document frequency matrix)
        
        n_components: Number of clusters.
        '''
        
        def __init__(self,data,n_components):
            self.data = data
            self.n_components = n_components

        def cluster_(self):
            '''
            returns: variance explained, components, array(lsa data)
            '''
            start_time = time.time()
            # Singular vector decomposition
            svd_ = TruncatedSVD(n_components=self.n_components)
            # pipelining 
            lsa_ = make_pipeline(svd_, Normalizer(copy=False))
            # latent semantic analysis
            X_lsa = lsa_.fit_transform(self.data)
            # variance explained
            explained_variance = svd_.explained_variance_ratio_.sum()
            print('time elapsed:', time.time()-start_time,'sec')
            return explained_variance,svd_.components_,X_lsa

        def concepts_(self,components,feature_names):
            start_time = time.time()
            # for each component
            for i,comp in enumerate(components):
                # feature names
                terms_in_comp = zip(feature_names,comp)
                # sorting most frequents 10 terms in descending order
                sortedterms = sorted(terms_in_comp,key = lambda x: x[1],reverse= True) [:10]
                print('concept:',i)
                # for each concept print sorted terms 
                for term in sortedterms:
                    print(term)
                # new line after each concept
                print (" ")
            print('time elapsed:', time.time()-start_time,'sec')
            
    class Kmeans(object):
        def __init__(self):
            pass
        def cluster(self,X,n_clusters):
            self.X = X
            self.n_clusters = n_clusters
            
            start_time = time.time()
            kmeans_init = KMeans(n_clusters= n_clusters,n_jobs= -1,precompute_distances=True)
            clus = kmeans_init.fit(self.X)
            if type(self.X) == np.ndarray:
                pred = clus.predict(self.X)
                print('time elapsed:', time.time()-start_time,'sec')
                return pred
            else: 
                print('time elapsed:', time.time()-start_time,'sec')
                return (clus.labels_)
