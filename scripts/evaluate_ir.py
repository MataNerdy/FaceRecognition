"""Evaluate trained embeddings with Identification Rate / TPR@FPR.

See notebooks/4_Identification_Rate_Metric_Clean.ipynb for the full experiment.
"""
import json

EXPERIMENT_RESULTS = {
    "CE": {
        0.50: {"tpr": 0.8678, "threshold": 0.9655},
        0.20: {"tpr": 0.5805, "threshold": 0.9782},
        0.10: {"tpr": 0.3858, "threshold": 0.9835},
        0.05: {"tpr": 0.2431, "threshold": 0.9871},
    },
    "ArcFace": {
        0.50: {"tpr": 0.6393, "threshold": 0.6294},
        0.20: {"tpr": 0.2936, "threshold": 0.8009},
        0.10: {"tpr": 0.1677, "threshold": 0.8634},
        0.05: {"tpr": 0.0976, "threshold": 0.9003},
    },
    "Triplet": {
        0.50: {"tpr": 0.7215, "threshold": 0.8385},
        0.20: {"tpr": 0.3877, "threshold": 0.9443},
        0.10: {"tpr": 0.2285, "threshold": 0.9688},
        0.05: {"tpr": 0.1333, "threshold": 0.9802},
    },
    "CE+ArcFace": {
        0.50: {"tpr": 0.7172, "threshold": 0.4576},
        0.20: {"tpr": 0.3668, "threshold": 0.6828},
        0.10: {"tpr": 0.2127, "threshold": 0.7635},
        0.05: {"tpr": 0.1282, "threshold": 0.8135},
    },
    "ArcFace+Triplet": {
        0.50: {"tpr": 0.7192, "threshold": 0.3941},
        0.20: {"tpr": 0.3587, "threshold": 0.6227},
        0.10: {"tpr": 0.1991, "threshold": 0.7176},
        0.05: {"tpr": 0.1112, "threshold": 0.7804},
    },
}

if __name__ == "__main__":
    print(json.dumps(EXPERIMENT_RESULTS, indent=2, ensure_ascii=False))
