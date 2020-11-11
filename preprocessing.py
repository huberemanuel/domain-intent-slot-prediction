import os
import pandas as pd
import numpy as np
import json
from collections import OrderedDict

dataset_path = os.path.join("data", "MULTIWOZ2 2")
with open(os.path.join(dataset_path, "data.json")) as f:
    dialog_data = json.load(f)

with open(os.path.join(dataset_path, "dialogue_acts.json")) as f:
    act_data = json.load(f)


df = pd.DataFrame(columns=["utterance", "domain", "intent", "slot", ])
data_lst = []

for act_key in act_data:
    dialogue_log = dialog_data[f"{act_key}.json"]["log"]
    dialogue_acts = OrderedDict(sorted(act_data[act_key].items(), key=lambda t: t[0]))
    for i in range(int(len(dialogue_log)/2)):
        
        # Ignoring extra unnecessary tags (e.g No Annotation)
        if i + 1 >= len(dialogue_acts):
            continue

        for act in dialogue_acts[str(i+1)]:
            if not "-" in act:
                continue
            domain, intent = act.split("-")

            for slot in dialogue_acts[str(i+1)][act]:
                data_lst.append({
                    "utterance": dialogue_log[i*2+1]["text"], 
                    "domain": domain.lower(), 
                    "intent": intent.lower(), 
                    "slot": slot[0].lower(),
                    "dis": "{}-{}-{}".format(domain, intent, slot[0]).lower()
                })
    system_text = dialog_data[f"{act_key}.json"]["log"]

df = df.append(data_lst)
df = df.drop(df[df.duplicated()].index)
df.to_csv(os.path.join("data", "train.csv"), index=False)
