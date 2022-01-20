import pandas as pd
import json
import pickle


# Read file pkl
print ("...Reading file.json")
with open("images/fileimg_label.json",'r') as json_f:
    data = json.load(json_f)
    # Convert from JSON to PICKLE
    print ("...Converting: file.json -> file.pkl")
    with open("images/file.pkl",'wb') as pkl_f:
      pickle.dump(data, pkl_f)

# Read file pkl
print ("...Reading file.pkl")
pkl = pd.read_pickle("images/file.pkl")
print(pkl)

print(pd.read_pickle("apps/fer_model/options.pkl"))

print ("...Done")