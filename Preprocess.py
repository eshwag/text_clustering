import time
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
class Preprocessing(object):
    
    def __init__(self):
        pass
    
    def text(self,X):
        ''' Basic text Preprocessing (Removing Url's, Special/Single characters,
        Stop words and Lemmatizing).
        
        Parameters:
        ----------
        X: A List containing data to be preprocessed
        
        returns: List (preprocessed)'''
        
        self.X = X
        start_time = time.time()
        if type(self.X) == list:
            # Converting into lower case
            lower_case = [element.lower() for element in self.X]
            # Removing url's 
            Url_removed = [re.sub('[^ ]+\.[^ ]+','',element) for element in lower_case]
            # Remving special characters
            SpecialChar_removed = [' '.join(re.findall(r'[a-z]+',line)) for line in Url_removed]
            # Removing single characters
            SingleChar_removed = [re.sub(r"\b[a-zA-Z]\b","",element) for element in SpecialChar_removed]
            # Removing stop words
            stop = stopwords.words('english')
            # Removing few words from stop words as they have some importance
            stop.remove('not')
            stop.remove('down')
            # Appending domain specific stop words 
            stop.extend(['kindly','please','perfmon','alarm','could','va','diag'])
            SingleChar_removed = pd.DataFrame(SingleChar_removed,columns=['samp'])
            StopWords_removed = SingleChar_removed.samp.apply(lambda x:' '.join(element for element in x.split() if element not in stop))
            # Lemmatizing
            Lemmatized = [WordNetLemmatizer().lemmatize(element) for element in list(StopWords_removed)]
            print('time elapsed:', time.time()-start_time,'sec')
        else:
            raise ValueError('Pass a List')
        return Lemmatized
    
    def text_vect(self,data,min_df = 2,max_df = 0.95,stop_words = {'english'},ngram_range = (1,3)):
        ''' Converts text to vectors using Tfidf
        Parameters:
        -----------
        data: List 
            
        min_df: float in range [0.0, 1.0] or int, default=1
                When building the vocabulary ignore terms that have a document
                frequency strictly lower than the given threshold. This value is also
                called cut-off in the literature.
                If float, the parameter represents a proportion of documents, integer
                absolute counts.
                
        max_df: float in range [0.0, 1.0] or int, default=1.0
                When building the vocabulary ignore terms that have a document
                frequency strictly higher than the given threshold (corpus-specific
                stop words).
                If float, the parameter represents a proportion of documents, integer
                absolute counts.
                
        stop_words : string {'english'}, list, or None (default)
                    If a string, it is passed to _check_stop_list and the appropriate stop
                    list is returned. 'english' is currently the only supported string
                    value.
                    
        ngram_range: tuple (min_n, max_n)
                    The lower and upper boundary of the range of n-values for different
                    n-grams to be extracted. All values of n such that min_n <= n <= max_n
                    will be used.
                    
        returns: Tfidf matrix and feture names'''
        
        start_time = time.time()
        # calling vectorizer
        vect = TfidfVectorizer(min_df= min_df,max_df= max_df,stop_words= stop_words,ngram_range=ngram_range)
        # tfidf matrix
        tfidf_matrix = vect.fit_transform(data)
        # feature names
        feature_names = vect.get_feature_names()
        print('time elapsed:', time.time()-start_time,'sec')
        return tfidf_matrix,feature_names