import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
from torch.nn import functional as F
from sklearn.metrics import classification_report
import os
from transformers import RobertaTokenizer, RobertaForTokenClassification


# --- Config (override via environment variables in Colab/cluster) ---
MODEL_NAME = os.environ.get("MODEL_NAME", "FacebookAI/roberta-large")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "checkpoints/st1_roberta")
TRAIN_FILE = os.environ.get("TRAIN_FILE", "data/Combined_dataset_CommonSense+News_Data/combined.csv")
DEV_FILE   = os.environ.get("DEV_FILE",   "data/News_data/dev.csv")
TEST_FILE  = os.environ.get("TEST_FILE",  "data/Test_dataset/test.csv")
SEED = int(os.environ.get("SEED", 42))
# Optional caps (mainly for sanity runs); empty string = use all rows
MAX_TRAIN = int(os.environ["MAX_TRAIN"]) if os.environ.get("MAX_TRAIN") else None
MAX_DEV   = int(os.environ["MAX_DEV"])   if os.environ.get("MAX_DEV")   else None
MAX_TEST  = int(os.environ["MAX_TEST"])  if os.environ.get("MAX_TEST")  else None
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load datasets (shuffle for reproducibility; cap if MAX_* is set)
def _load(path, cap):
    df = pd.read_csv(path)
    # Coerce to plain Python strings and drop any rows missing text/relation.
    # Newer pandas uses StringDtype which datasets.Dataset.from_pandas can
    # surface as a non-list type to the tokenizer; astype(object) avoids that.
    df = df.dropna(subset=['text', 'relation'])
    df['text']     = df['text'].astype(object).map(str)
    df['relation'] = df['relation'].astype(object).map(str)
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return df.iloc[:cap] if cap else df

df_train = _load(TRAIN_FILE, MAX_TRAIN)
df_dev   = _load(DEV_FILE,   MAX_DEV)
df_test  = _load(TEST_FILE,  MAX_TEST)
print(f"Loaded: train={len(df_train)} dev={len(df_dev)} test={len(df_test)}")

# Ensure correct column names
df_train = df_train.rename(columns={"text": "text", "relation": "label"})
df_dev = df_dev.rename(columns={"text": "text", "relation": "label"})
df_test = df_test.rename(columns={"text": "text", "relation": "label"})

# Encode labels
label2id = {label: idx for idx, label in enumerate(pd.concat([df_train, df_dev, df_test])['label'].unique())}
print("Original labels:", df_train["label"].unique())
print("Encoded labels:", df_train["label"].map(label2id).unique())
id2label = {idx: label for label, idx in label2id.items()}
df_train['label'] = df_train['label'].map(label2id)
df_dev['label'] = df_dev['label'].map(label2id)
df_test['label'] = df_test['label'].map(label2id)


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def _as_str_list(x):
    # Defensive: datasets >= 3 can return non-list types (Arrow array, numpy
    # array) from `dataset[col]`/`select()[col]` that the tokenizer rejects
    # with "text input must be of type str ...".
    return [str(t) for t in x]

def tokenize_function(examples):
    tokenized_inputs = tokenizer(_as_str_list(examples["text"]), padding="max_length", truncation=True)
    tokenized_inputs["label"] = examples["label"]
    return tokenized_inputs



dataset = {
    "train": Dataset.from_pandas(df_train),
    "validation": Dataset.from_pandas(df_dev),
    "test": Dataset.from_pandas(df_test),
}

tokenized_datasets = {split: dataset[split].map(tokenize_function, batched=True) for split in dataset}



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(label2id), id2label=id2label, label2id=label2id
).to(device)

LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 2e-5))
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

train_dataset = tokenized_datasets["train"]
validation_dataset = tokenized_datasets["validation"]

epochs = int(os.environ.get("EPOCHS", 10))
batch_size = int(os.environ.get("BATCH_SIZE", 8))
best_val_loss = float("inf")
best_model_path = OUTPUT_DIR


from tqdm import tqdm

for epoch in range(epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(range(0, len(train_dataset), batch_size), desc=f"Epoch {epoch + 1}", leave=False)

    for i in progress_bar:
        batch = train_dataset.select(range(i, min(i + batch_size, len(train_dataset))))
        inputs = tokenizer(_as_str_list(batch["text"]), padding=True, truncation=True, return_tensors="pt").to(device)
        labels = torch.tensor(batch["label"], dtype=torch.long).to(device)

        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    avg_train_loss = total_loss / len(train_dataset)
    print(f"Epoch {epoch + 1}: Training Loss = {avg_train_loss}")

    model.eval()
    val_loss = 0
    for i in range(0, len(validation_dataset), batch_size):
        batch = validation_dataset.select(range(i, min(i + batch_size, len(validation_dataset))))
        inputs = tokenizer(_as_str_list(batch["text"]), padding=True, truncation=True, return_tensors="pt").to(device)
        labels = torch.tensor(batch["label"]).to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=labels)
            val_loss += outputs.loss.item()

    avg_val_loss = val_loss / len(validation_dataset)
    print(f"Epoch {epoch + 1}: Validation Loss = {avg_val_loss}")


    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        print("Saving new best model...")
        model.save_pretrained(best_model_path)
        tokenizer.save_pretrained(best_model_path)

# Free GPU memory
# del model
# torch.cuda.empty_cache()

# Reload best model for testing
print("Loading best model for evaluation...")
model = AutoModelForSequenceClassification.from_pretrained(best_model_path).to(device)
model.eval()


test_texts = dataset["test"]["text"]
test_labels = torch.tensor(dataset["test"]["label"]).to(device)

batch_size = 8
all_preds = []
all_labels = []

for i in range(0, len(test_texts), batch_size):
    batch_texts = test_texts[i : i + batch_size]
    batch_labels = test_labels[i : i + batch_size]

    inputs = tokenizer(_as_str_list(batch_texts), padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=-1)

    all_preds.extend(preds.cpu().numpy())
    all_labels.extend(batch_labels.cpu().numpy())


report = classification_report(all_labels, all_preds, target_names=[id2label[i] for i in range(len(id2label))])
print(report)


with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

# # Free GPU memory after evaluation
# del model
# torch.cuda.empty_cache()
