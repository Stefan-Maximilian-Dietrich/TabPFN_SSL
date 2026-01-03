import os
import csv
import time
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

import data 
import functions 
import decision 
import experiments

def run_one_seed(seed: int, jobid: str, rank: int, results_dir: str, experiment_nr: int):
    """
    Führt einen TabPFN-Lauf für einen Seed aus und speichert Accuracy & AUC als CSV.

    seed       : random_state für train_test_split
    jobid      : Slurm-Job-ID (nur für Logging)
    rank       : globaler Rank (0..WORLD_SIZE-1)
    results_dir: Zielordner für CSV
    """

    os.makedirs(results_dir, exist_ok=True)


    # 6) Ergebnisse als CSV speichern
    exp = experiments.Experiments[experiment_nr]
    n = exp["n"]    
    m = exp["m"]
    Data = exp["Data"]
    Sampler = exp["Sampler"]
    Evaluation = exp["Evaluation"]
    Classifier = exp["Classifier"]
    Decision = exp["Decision"]
    Predict = exp["Predict"]

    np.random.seed(seed)
    ssl = functions.SSL( n, m, Data, Sampler, Evaluation, Classifier, Decision, Predict)
    result = ssl.run(seed)
    result_sl = ssl.run_SL(seed)

    #Gener Path
    experiment_dir_1lv = os.path.join(results_dir, f"{Data.name}_{n}")
    os.makedirs(experiment_dir_1lv, exist_ok=True)
    experiment_dir_2lv = os.path.join(experiment_dir_1lv, f"unlabled_{m}")
    os.makedirs(experiment_dir_2lv, exist_ok=True)
    classifyer_dir = os.path.join(experiment_dir_2lv, f"classifyer_{Classifier.name}")
    os.makedirs(classifyer_dir, exist_ok=True)

    #SSL Method Path
    decision_dir = os.path.join(classifyer_dir, f"decision_{Decision(Decision).name}")
    os.makedirs(decision_dir, exist_ok=True)
    csv_path = os.path.join(decision_dir, f"ID_{seed}.csv")
    functions.save_confusion_matrices_long(result=result, csv_path=csv_path, jobid=jobid, rank=rank, seed=seed,)

    #SL Method Path 
    decision_dir_sl = os.path.join(classifyer_dir, f"decision_supervised")
    os.makedirs(decision_dir_sl, exist_ok=True)
    csv_path_sl = os.path.join(decision_dir_sl, f"ID_{seed}.csv")
    functions.save_confusion_matrices_long(result=result_sl, csv_path=csv_path_sl, jobid=jobid, rank=rank, seed=seed)

    print(csv_path)

    ### speichermethode finden conf mat als csv 

    print(f"[Rank: {rank}] Seed: {seed} DONE")
    print(f"path: {csv_path}")
