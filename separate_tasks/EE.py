import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForTokenClassification
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score
import os

# --- Config (override via environment variables in Colab/cluster) ---
MODEL_NAME    = os.environ.get("MODEL_NAME", "dslim/bert-large-NER")
OUTPUT_DIR    = os.environ.get("OUTPUT_DIR", "checkpoints/st2_bert_ner")
TRAIN_FILE    = os.environ.get("TRAIN_FILE", "data/Combined_dataset_CommonSense+News_Data/combined.csv")
DEV_FILE      = os.environ.get("DEV_FILE",   "data/News_data/dev.csv")
TEST_FILE     = os.environ.get("TEST_FILE",  "data/Test_dataset/test.csv")
MAX_LEN       = int(os.environ.get("MAX_LEN", 128))
BATCH_SIZE    = int(os.environ.get("BATCH_SIZE", 16))
EPOCHS        = int(os.environ.get("EPOCHS", 10))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 2e-5))
SEED          = int(os.environ.get("SEED", 42))
MAX_TRAIN     = int(os.environ["MAX_TRAIN"]) if os.environ.get("MAX_TRAIN") else None
MAX_DEV       = int(os.environ["MAX_DEV"])   if os.environ.get("MAX_DEV")   else None
MAX_TEST      = int(os.environ["MAX_TEST"])  if os.environ.get("MAX_TEST")  else None
os.makedirs(OUTPUT_DIR, exist_ok=True)

BIO_LABELS   = {"O": 0, "B-SUBJ": 1, "I-SUBJ": 2, "B-OBJ": 3, "I-OBJ": 4}
ID2LABEL_BIO = {v: k for k, v in BIO_LABELS.items()}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# Dataset class
class BIOTaggingDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
        self.texts = self.data['text'].tolist()
        self.subjects = self.data['subject'].tolist()
        self.objects = self.data['object'].tolist()
        self.relations = self.data['relation'].tolist()

    def bio_tagging(self, text, subject, obj, relation):
        tokens = tokenizer.tokenize(text)
        labels = ["O"] * len(tokens)

        if relation != "no_relation":
            for ent, tag in [(subject, "SUBJ"), (obj, "OBJ")]:
                if isinstance(ent, str) and ent.strip():  # Ensure ent is a non-empty string
                    ent_tokens = tokenizer.tokenize(ent)
                    for i in range(len(tokens) - len(ent_tokens) + 1):
                        if tokens[i:i + len(ent_tokens)] == ent_tokens:
                            labels[i] = f"B-{tag}"
                            for j in range(1, len(ent_tokens)):
                                labels[i + j] = f"I-{tag}"
                            break

        if len(tokens) > MAX_LEN:
            tokens = tokens[:MAX_LEN]
            labels = labels[:MAX_LEN]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        label_ids = [BIO_LABELS[label] for label in labels]

        padding_length = MAX_LEN - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        label_ids += [BIO_LABELS["O"]] * padding_length

        return input_ids, attention_mask, label_ids

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        input_ids, attention_mask, label_ids = self.bio_tagging(self.texts[idx], self.subjects[idx], self.objects[idx],
                                                                self.relations[idx])
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }


# Model definition
class NERModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super(NERModel, self).__init__()
        self.bert = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=ID2LABEL_BIO,
            label2id=BIO_LABELS,
            ignore_mismatched_sizes=True,
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits


# Load dataset (shuffle for reproducibility; cap if MAX_* is set; coerce to str)
def load_data(filepath, cap=None):
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['text', 'subject', 'object', 'relation'])
    for col in ('text', 'subject', 'object', 'relation'):
        df[col] = df[col].astype(object).map(str)
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    if cap:
        df = df.iloc[:cap]
    return df


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), \
                                                batch['labels'].to(device)
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    f1 = f1_score(all_labels, all_preds, average='macro')
    report = classification_report(all_labels, all_preds, target_names=BIO_LABELS.keys())
    print(report)
    return f1


def train():

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_dataset = BIOTaggingDataset(load_data(TRAIN_FILE, cap=MAX_TRAIN))
        dev_dataset   = BIOTaggingDataset(load_data(DEV_FILE,   cap=MAX_DEV))
        test_dataset  = BIOTaggingDataset(load_data(TEST_FILE,  cap=MAX_TEST))
        print(f"Loaded: train={len(train_dataset)} dev={len(dev_dataset)} test={len(test_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = NERModel(MODEL_NAME, len(BIO_LABELS)).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        best_f1 = 0.0

        model.train()
        for epoch in range(EPOCHS):
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False)
            for batch in progress_bar:
                input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), \
                                                    batch['labels'].to(device)

                optimizer.zero_grad()
                logits = model(input_ids, attention_mask)
                loss = criterion(logits.view(-1, len(BIO_LABELS)), labels.view(-1))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

            print("Evaluating on Dev Set:")
            f1 = evaluate(model, dev_loader, device)
            print(f1)

            if f1 > best_f1:
                best_f1 = f1
                # Save as HF folder (easy resume via from_pretrained later)
                model.bert.save_pretrained(OUTPUT_DIR)
                tokenizer.save_pretrained(OUTPUT_DIR)
                print(f"Saved best model to {OUTPUT_DIR} (dev macro F1={best_f1:.4f})")

        print("Testing on Test Set:")
        evaluate(model, test_loader, device)




if __name__ == "__main__":
    train()

