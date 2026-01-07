import modules.data as ds
import scripts.functions as fun
import modules.classifier as cl
import modules.decision as dec

print("Toy Examples")
DATASETS = [
    ds.Spirals()
]

CLASSIFIERS = [
    cl.NaiveBayesClassifier(),
    cl.MultinomialLogitClassifier(),
    cl.SmallNNClassifier(),
    cl.SVMClassifier(),
    cl.RandomForestCls(),
    cl.GradientBoostingCls(),
    cl.DecisionTreeCls(),
    cl.KNNClassifier(),
    cl.TabPfnClassifier(),

]

DECISIONS = [
    dec.SSL_prob,
]

UNLABLED = [
    4,
    8
]

LABLED = [
    8,
    10,
    20,
    30,
    40
]

Experiments = [
    {
        "n": n,
        "m": m,
        "Data": data_obj,
        "Sampler": fun.upsample,
        "Evaluation": fun.confusion,
        "Classifier": clf_obj,
        "Decision": decision_fn,
        "Predict": fun.predictor,
    }
    for n in LABLED
    for m in UNLABLED
    for data_obj in DATASETS
    for clf_obj in CLASSIFIERS
    for decision_fn in DECISIONS
    
]

print("Anzahl Experimente:", len(Experiments))

