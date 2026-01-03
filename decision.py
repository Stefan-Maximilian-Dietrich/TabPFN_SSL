from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
import pandas as pd
#from huggingface_hub import login
import numpy as np


#login(token="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")

class maximalPPP: #checked
    def __init__(self, Classifier):
        self.name = "maximalPPP"

    def ppp(self, sets):
        model = TabPFNClassifier()
        X = sets.drop(columns=["target"])
        y = sets["target"]
        model.fit(X, y)
        prob = model.predict_proba(X)  
        correct = prob[np.arange(len(prob)), y]

        return np.prod(correct)

    def __call__(self, labeled, pseudo):
        all_sets = [pd.concat([labeled, pseudo.iloc[[i]]], ignore_index=True)
                for i in range(len(pseudo))]
        PPP = [self.ppp(ds) for ds in all_sets]
        maximal = int(np.argmax(PPP))
        return maximal

class SSL_prob: 
    def __init__(self, Classifier):
        self.name = "maximalProb"
        self.model = Classifier

    def prob(self, labeled, pseudo):
        self.model.fit(labeled)
        self.model.predict_proba(pseudo)
        
    def __call__(self, labeled, pseudo):
        proba = self.prob(labeled, pseudo)
        max_proba = np.max(proba, axis=0)
        winner = int(np.argmax(max_proba))
        return winner
        
    
class SSL_confidence:
    def __init__(self, classifier):
        self.name = "maximalConfidence" 
        self.model = classifier

    def prob(self, labeled, pseudo):
        self.model.fit(labeled)
        proba_all = self.model.predict_proba(pseudo)
        p_class1 = proba_all[:, 1]
        return p_class1

    def __call__(self, labeled, pseudo):
        p = self.prob(labeled, pseudo)

        abs_confidence = np.where(p > 0.5, 1.0 - p, p)
        winner = int(np.argmin(abs_confidence))
        return winner