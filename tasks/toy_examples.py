import modules.data as ds
import scripts.functions as fun
import modules.classifier as cl
import modules.decision as dec

print("Toy Examples")
DATASETS = [
    ds.Spirals(),
    ds.Bank()
]

CLASSIFIERS = [
    cl.SVMClassifier(),
    cl.KNNClassifier(),

]

DECISIONS = [
    dec.SSL_prob,
    dec.maximalPPP,
    dec.SSL_confidence,
]

UNLABLED = [
    4,
    12
]

LABLED = [
    4,
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

print("Number of experiments:", len(Experiments))

