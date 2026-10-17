import datasets;

ds = datasets.load_dataset("holodata/sensai");
flagged = ds["train"].to_pandas();

flagged['label'] = flagged['label'].map({
	'hidden': 1,
	'deleted': 1,
	'nonflagged': 0
})

flagged['body'] = flagged['body'].str.lower() # normalise the text
flagged = flagged.drop_duplicates(subset=['body']) # drop duplicates
flagged['body'] = flagged['body'].str.strip() # remove unecessary whitespace
flagged = flagged.dropna() # remove empty cells

flagged.to_json("./local/dataset.json");
