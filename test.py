import torch;
from typing import cast;
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification;

print("Loading model...");

device = torch.device("cpu");
tokenizer = cast(DistilBertTokenizerFast, DistilBertTokenizerFast.from_pretrained("./local/model"));
model = cast(DistilBertForSequenceClassification, DistilBertForSequenceClassification.from_pretrained("./local/model"));
model.to(device);
model.eval();

def is_safe(message: str) -> bool:
	inputs = tokenizer(message, return_tensors="pt", truncation=True, max_length=256);
	inputs = { k : v.to(device) for k, v in inputs.items() };

	with torch.no_grad():
		outputs = model(**inputs);
		result = torch.argmax(outputs.logits, dim=1).item();

	return result == 1;

print("Testing model...");

incoming_messages = [
	"Hey, what time does the raid start?",
	"You are complete trash, leave the game.",
	"Good job team!"
]

for msg in incoming_messages:
	safe = is_safe(msg)
	status = "ALLOW" if safe else "BLOCK"
	print(f"[{status}] Message: '{msg}' (Safe: {safe})")
