import numpy as np
import nltk
import string
from nltk.corpus import stopwords
import gensim
import time



stops = set(stopwords.words("english"))
punct = set(string.punctuation)

class Word2vecExtractor:

    def __init__(self, w2vecmodel):
        #self.w2vecmodel=gensim.models.Word2Vec.load_word2vec_format(w2vecmodel, binary=binary)
        self.w2vecmodel = gensim.models.KeyedVectors.load_word2vec_format(w2vecmodel)

        #self.w2vecmodel = w2vecmodel

    def sen2vec(self,sentence):
        words = [word for word in nltk.word_tokenize(sentence) if word not in stops and word not in punct]
        res = np.zeros(self.w2vecmodel.vector_size)
        count = 0
        for word in words:
            if word in self.w2vecmodel:
                count += 1
                res += self.w2vecmodel[word]

        if count != 0:
            res /= count

        return res 

    def doc2vec(self, doc):
        count = 0    
        res = np.zeros(self.w2vecmodel.vector_size)
        for sentence in nltk.sent_tokenize(doc):
            for word in nltk.word_tokenize(sentence):
                if((word not in stops) and (word not in punct)):
                    if word in self.w2vecmodel:
                        count += 1
                        res += self.w2vecmodel[word]

        if count != 0:
            res /= count

        return res 

    def get_doc2vec_feature_dict(self, doc):  
        vec = self.doc2vec(doc)

        number_w2vec=vec.size    
        feature_dict = {}

        for i in range(0, number_w2vec):
            feature_dict.update({"Word2Vecfeature_"+ str(i):vec[i]})
           
        return feature_dict
    
    def word2v(self, word):
        res = np.zeros(self.w2vecmodel.vector_size)
        if word in self.w2vecmodel:
            res += self.w2vecmodel[word]
        return res



if __name__ == "__main__":
    t0 = time.time()
    glove_w2v_file = "data/glove-w2v.txt"
    W2vecextractor = Word2vecExtractor(glove_w2v_file)


    t1 = time.time()
    print("done loading word vectors: ", (t1 - t0) / 60.0)
    doc = "A fisherman was catching fish by the sea. A monkey saw him, and wanted to imitate what he was doing. The man went away into a little cave to take a rest, leaving his net on the beach. The monkey came and grabbed the net, thinking that he too would go fishing. But since he didn't know anything about it and had not had any training, the monkey got tangled up in the net, fell into the sea, and was drowned. The fisherman seized the monkey when he was already done for and said, 'You wretched creature! Your lack of judgment and stupid behaviour has cost you your life!'"

    sent_vec = W2vecextractor.doc2vec(doc)
    print(sent_vec)

    t2 = time.time()
    print("execution time:", (t1 - t0) / 60.0)




