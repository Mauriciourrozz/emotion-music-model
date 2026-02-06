import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

"""
Tokenizes processed data for use with Transformers models.
"""
# Load processed CSVs
train_df = pd.read_csv("data/train_processed.csv")
dev_df = pd.read_csv("data/dev_processed.csv")
test_df = pd.read_csv("data/test_processed.csv")

"""Map GO emotions to numeric labels"""
# Label mapping
label2id = {
    "joy": 0,
    "anger": 1,
    "sadness": 2,
    "fear": 3
}

# Convert emotion → label id
def encode_labels(df):
    df = df.copy()
    df["label"] = df["emotion"].map(label2id)
    return df

"""Apply label encoding"""
train_df = encode_labels(train_df)
dev_df = encode_labels(dev_df)
test_df = encode_labels(test_df)

# Convert to HF Dataset
train_dataset = Dataset.from_pandas(train_df)
dev_dataset = Dataset.from_pandas(dev_df)
test_dataset = Dataset.from_pandas(test_df)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenization function
def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

# Apply tokenization
train_dataset = train_dataset.map(tokenize, batched=True)
dev_dataset = dev_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# Set format for PyTorch
train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

dev_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

test_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

train_dataset.save_to_disk("data/tokenized_train")
dev_dataset.save_to_disk("data/tokenized_dev")
test_dataset.save_to_disk("data/tokenized_test")

print("Tokenización completa")
print(train_dataset[0])
