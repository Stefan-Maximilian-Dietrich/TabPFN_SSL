import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] 
sys.path.insert(0, str(ROOT))               

import os
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

import scripts.functions  as functions

import importlib



def run_one_seed(task_path: str, seed: int, jobid: str, rank: int, results_dir: str, experiment_nr: int):
    os.makedirs(results_dir, exist_ok=True)

    module_name = task_path.replace("/", ".").replace(".py", "")

    task_module = importlib.import_module(module_name)

    # Zugriff auf Funktionen
    exp = getattr(task_module, "Experiments")

    # 6) Ergebnisse als CSV speichern
    exp = task_module.Experiments[experiment_nr]
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
    print("SSL gestartet")
    result = ssl.run(seed)
    result_sl = ssl.run_SL(seed)
    print("Ende")

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
