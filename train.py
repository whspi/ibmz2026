import torch;
import pandas;
from typing import cast;
from transformers import (
	DataCollatorWithPadding,
	DistilBertTokenizerFast,
	DistilBertForSequenceClassification
);

print("Loading base model...");

device = torch.device("cpu");
tokenizer = cast(DistilBertTokenizerFast, DistilBertTokenizerFast.from_pretrained("./local/base-model"));
model = cast(DistilBertForSequenceClassification, DistilBertForSequenceClassification.from_pretrained("./local/base-model"));
model.to(device);

class ModerationDataset(torch.utils.data.Dataset):
	def __init__(self, file):
		data = pandas.read_json(file);
		body = data["body"];

		self.len = len(body);
		self.texts = body.iloc;
		self.labels = data["label"].iloc;

	def __len__(self):
		return self.len;

	def __getitem__(self, idx):
		text = self.texts[idx];
		label = self.labels[idx];

		encoding = tokenizer(text, truncation=True, max_length=256);
		encoding["label"] = label;
		return encoding;

print("Loading dataset...");

train_dataset = ModerationDataset("./local/dataset.json");
train_loader = torch.utils.data.DataLoader(train_dataset,
	collate_fn = DataCollatorWithPadding(tokenizer=tokenizer),
	batch_size = 16,
	shuffle = True
);

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5);
model.train();
print("Training model...");

for epoch in range(3):
	for batch in train_loader:
		optimizer.zero_grad();
		input_ids = batch["input_ids"].to(device);
		attention_mask = batch["attention_mask"].to(device);
		labels = batch["labels"].to(device);

		outputs = model(input_ids, attention_mask=attention_mask, labels=labels);
		outputs.loss.backward();
		optimizer.step();

	print("Epoch " + str(epoch) + " finished.");

print("Saving model...");
model.save_pretrained("./local/model");
tokenizer.save_pretrained("./local/model");
