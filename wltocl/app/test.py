# test_predict_numpy.py
import os, json, numpy as np
MODEL = r".\model\logreg_model.json"
if not os.path.exists(MODEL):
    print("No numpy model at", MODEL); raise SystemExit(1)
with open(MODEL) as f:
    saved = json.load(f)
print("feature names:", saved["feature_names"][:10])
w = np.array(saved["weights"])
b = saved["bias"]
print("weights len:", len(w), "bias:", b)
# quick numeric test
x = np.array([0.0]*len(w))
z = x.dot(w) + b
p = 1/(1+np.exp(-z))
print("sanity prob %:", p*100)
