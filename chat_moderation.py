import pandas as pd
import json
from datasets import load_dataset

ds = load_dataset("holodata/sensai")
flagged = ds['train'].to_pandas()

# map 'hidden' and 'deleted' to 1 (delete), and 'nonflagged' to 0 (keep)
flagged['label'] = flagged['label'].map({
    'hidden': 1,
    'deleted': 1,
    'nonflagged': 0
})


flagged['body'] = flagged['body'].str.lower() # normalise the text
flagged = flagged.drop_duplicates(subset=['body']) # drop duplicates 
flagged['body'] = flagged['body'].str.strip() # remove unecessary whitespace 
flagged = flagged.dropna() # remove empty cells

flagged["body"].to_json("text.json", orient="values")
flagged["label"].to_json("label.json", orient="values")

