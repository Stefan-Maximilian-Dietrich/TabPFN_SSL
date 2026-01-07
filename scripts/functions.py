import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import os
import csv

class upsample: #checked
    def __init__(self, n, m, Data):
        self.n = n 
        self.m = m
        self.data = Data

    def __call__(self, seed):
        df = self.data.copy()
        n_labeled = self.n  
        n_unlabeled = self.m 
        classes = df["target"].unique()
        k = 1  # mindestens 1 pro Klasse
        labeled_idx = []
        np.random.seed(seed)
        for c in classes:
            idx = df.index[df["target"] == c].to_numpy()
            chosen = np.random.choice(idx, size=k, replace=False)
            labeled_idx.extend(chosen)

        labeled_idx = np.array(labeled_idx)
        remaining = np.setdiff1d(df.index, labeled_idx)

        n_missing = n_labeled - len(labeled_idx)
        if n_missing > 0:
            extra = np.random.choice(remaining, size=n_missing, replace=False)
            labeled_idx = np.concatenate([labeled_idx, extra])
            remaining = np.setdiff1d(remaining, extra)

        unlabeled_idx = np.random.choice(remaining, size=n_unlabeled, replace=False)
        test_idx = np.setdiff1d(remaining, unlabeled_idx)

        return (
            df.loc[labeled_idx].reset_index(drop=True),
            df.loc[unlabeled_idx].reset_index(drop=True),
            df.loc[test_idx].reset_index(drop=True)
        )

class predictor: #ckecked
    def __init__(self, Classifier):
        self.model = Classifier

    def __call__(self, labeled, unlabeled):
        self.model.fit(labeled)
        y_pseudo = self.model.predict(unlabeled)
        unlabeled.drop(columns=["target"]) #prevent lekage
        unlabeled["target"] = y_pseudo
        return unlabeled

class confusion: #checked 
    def __init__(self, classifier):
        self.model = classifier

    def __call__(self, labled, test):
        self.model.fit(labled)
        y_pred = self.model.predict(test)
        conf_mat = confusion_matrix(test["target"], y_pred)
        return conf_mat

class SSL: 
    def __init__(self, n, m, Data, Sampler, Evaluation, Classifier, Decision, Predict):
        self.n = n
        self.m = m

        self.data = Data
        self.sampel = Sampler(n, m, self.data())
        self.classifier = Classifier
        self.evaluation = Evaluation(Classifier)
        self.predict = Predict(Classifier)
        self.decision = Decision(Classifier)

    def split_data(self, train, pseudo, decision):
        train = pd.concat([train, pseudo.iloc[[decision]]], ignore_index=True)
        unlabeled = pseudo.drop(index=decision).reset_index(drop=True)
        return train, unlabeled

    def save(self, seed, result):
        full_path = os.path.join(self.path, str(seed))
        np.save(full_path, result)

    def run_SL(self, seed):
        labeled, unlabeled, test = self.sampel(seed)
        result = []
        result.append(self.evaluation(labeled, test))
        for _ in range(len(unlabeled)):
            result.append(self.evaluation(labeled, test))
        return result
        
    def run(self, seed):
        labeled, unlabeled, test = self.sampel(seed)
        result = []
        result.append(self.evaluation(labeled, test))

        while len(unlabeled) > 0:
            pseudo = self.predict(labeled, unlabeled)
            decision = self.decision(labeled, pseudo)
            labeled, unlabeled = self.split_data(labeled, pseudo, decision)
            for _ in range(len([decision])):
                result.append(self.evaluation(labeled, test))

        return result


def save_confusion_matrices_long(result, csv_path, jobid, rank, seed):

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)


    conf_list = result # Liste von Matrizen

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "jobid",
                "rank",
                "seed",
                "cm_index",   # Index der Matrix in der Liste
                "true",       # wahrer Label
                "pred",       # vorhergesagter Label
                "count",      # Häufigkeit
            ],
        )
        writer.writeheader()

        for cm_index, cm in enumerate(conf_list):
            cm = np.asarray(cm)

            # cm.shape = (n_true, n_pred)
            n_true, n_pred = cm.shape

            for true_label in range(n_true):
                for pred_label in range(n_pred):
                    count = int(cm[true_label, pred_label])

                    writer.writerow(
                        {
                            "jobid": jobid,
                            "rank": rank,
                            "seed": seed,
                            "cm_index": cm_index,
                            "true": true_label,
                            "pred": pred_label,
                            "count": count,
                        }
                    )

