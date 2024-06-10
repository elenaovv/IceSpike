import os
from sklearn.metrics import f1_score, matthews_corrcoef
from transformers import (
    Trainer,
    TrainingArguments,
    AdamW,
    get_linear_schedule_with_warmup,
    get_scheduler,
    EarlyStoppingCallback,  # Ensure this is added
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    TrainerCallback,
    TrainerState, 
    TrainerControl
)
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from torchmetrics.classification import MatthewsCorrCoef

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

RANDOM_SEED = 42
EPOCHS = 4
LEARNING_RATE = 1e-6
BATCH_SIZE = 8

model_name = "mideind/IceBERT-ic3"
model_save_dir = "elenaovv/model_igc_full"

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)


def tokenize_data(data, tokenizer, max_len=256):
    print("Tokenizing data, number of samples:", len(data))
    tokenized_data = tokenizer(
        data.tolist(), padding="max_length", truncation=True, max_length=max_len
    )
    # Optionally print the first tokenized sample to check output
    print("Example of tokenized data:", tokenized_data['input_ids'][0])
    return tokenized_data

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

train = pd.read_csv('/users/home/elenao23/2_finetuning_fix/data/igc_2_3/train.txt', sep='\t', header=None, names=['text', 'label'])
test = pd.read_csv('/users/home/elenao23/2_finetuning_fix/data/igc_2_3/test.txt', sep='\t', header=None, names=['text', 'label'])
validation = pd.read_csv('/users/home/elenao23/2_finetuning_fix/data/igc_2_3/validation.txt', sep='\t', header=None, names=['text', 'label'])

# Print sample data to check loading and formatting
print("Sample training data:", train.head())
print("Sample test data:", test.head())
print("Sample validation data:", validation.head())

# Check label distribution
print("Training label distribution:\n", train['label'].value_counts())
print("Validation label distribution:\n", validation['label'].value_counts())
print("Test label distribution:\n", test['label'].value_counts())


def convert(sentiment):
    return 1 if sentiment == "positive" else 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ProcessDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels.iloc[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(p):
    logits, labels = p
    predictions = np.argmax(logits, axis=-1)
    return {
        "acc": (predictions == labels).mean(),
        "f1": f1_score(labels, predictions, average="weighted"),
        "mcc": matthews_corrcoef(labels, predictions)
    }


model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=4,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.to(device)

# Tokenize the datasets
train_data = tokenize_data(train['text'], tokenizer)
val_data = tokenize_data(validation['text'], tokenizer)
test_data = tokenize_data(test['text'], tokenizer)

train_dataset = ProcessDataset(train_data, train['label'])
val_dataset = ProcessDataset(val_data, validation['label'])
test_dataset = ProcessDataset(test_data, test['label'])

print("Checking dataset by fetching a few samples...")
for i in range(3):
    print(train_dataset[i])

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)

total_steps = len(train_dataset) * EPOCHS
print(total_steps)
scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

log_dir = os.path.join(parent_dir, "logs")

training_args = TrainingArguments(
    output_dir=os.path.join(parent_dir, "Models_igc_2_3/results/" + model_save_dir),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir=log_dir,
    load_best_model_at_end=True,
    learning_rate=LEARNING_RATE,
)

class PrintMetricsCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, metrics=None, **kwargs):
        print("Metrics at end of epoch", state.epoch, ":", metrics)

class ValidationCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        import random
        tokenizer = kwargs['tokenizer']
        dataset = kwargs['eval_dataset']
        model = kwargs['model']
        # Randomly pick a few samples from the validation set
        sample_indices = random.sample(range(len(dataset)), 3)
        for idx in sample_indices:
            example = dataset[idx]
            inputs = {key: example[key].unsqueeze(0).to(model.device) for key in ['input_ids', 'attention_mask']}
            labels = example['labels'].unsqueeze(0).to(model.device)
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)
            print(f"Input Text: {tokenizer.decode(example['input_ids'])}")
            print(f"Actual Label: {labels.item()}, Predicted Label: {predictions.item()}")


trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[
        #EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01),
        #ProgressCallback()
        #PrintMetricsCallback(),
        #ValidationCallback(), 
        #EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01)
    ],
    optimizers=(optimizer, scheduler),
    tokenizer=tokenizer,
)

# Print model configuration to confirm setup
print("Model configuration:", model.config)

trainer.train()

results = trainer.evaluate(test_dataset)
print("test results:", results)

model.save_pretrained(os.path.join(parent_dir, "Models_igc_2_3/" + model_save_dir))