pip install transformers datasets evaluate rouge_score accelerate

import transformers

transformers.__version__

"""# **Huuging Face Access Token with Read Permission**"""

from huggingface_hub import notebook_login

notebook_login()

"""# **Checking all datasets available in huggingface hub**

https://huggingface.co/docs/datasets/v1.2.0/loading_datasets.html

"""

from huggingface_hub import list_datasets

datasets_list = list_datasets()

num = 0
for dataset in datasets_list:
  print(dataset)

  if (num > 10):
    break

  num += 1

"""# **Loading Dataset**"""

from datasets import load_dataset

ai_medical_chatbot_ds = load_dataset("ruslanmv/ai-medical-chatbot")
ai_medical_chatbot_ds

"""Dataset dictionary can be seen. As the dataset is quite large, we would be taking a subset of training data only."""

ai_medical_chatbot_ds = load_dataset("ruslanmv/ai-medical-chatbot", split = "train[:70%]")

ai_medical_chatbot_ds

"""Subset data's dimensions can be checked"""

ai_medical_chatbot_ds.shape

"""dataset description, features, homepage can be obtained
https://huggingface.co/docs/datasets/v1.2.0/exploring.html
"""

print(ai_medical_chatbot_ds.description)

ai_medical_chatbot_ds.features

ai_medical_chatbot_ds.homepage

"""One row of example can be obtained"""

ai_medical_chatbot_ds[0]

"""Dataset is split into train and test sets"""

ai_medical_chatbot_ds = ai_medical_chatbot_ds.train_test_split(test_size = 0.2, seed = 42)

ai_medical_chatbot_ds

"""Text cleaning function is defined"""

def clean_txt(example):
    for txt in ["Description", "Patient", "Doctor"]:
       example[txt]  = example[txt].lower()
       example[txt]  = example[txt].replace("\\", "")
       example[txt]  = example[txt].replace("/", "")
       example[txt]  = example[txt].replace("\n", "")
       example[txt]  = example[txt].replace("``", "")
       example[txt]  = example[txt].replace('"', '')
       example[txt]  = example[txt].replace("--", "")

    return example

"""Both train and test sets are cleaned using 'map' method"""

cleaned_ai_medical_chatbot_ds = ai_medical_chatbot_ds.map(clean_txt)

cleaned_ai_medical_chatbot_ds

"""Taking a look at a raw example"""

print(cleaned_ai_medical_chatbot_ds["train"][7])

"""Now observing cleaned up version of the same raw data"""

cleaned_ai_medical_chatbot_ds["train"][7]

"""Similarly,both raw and cleaned up version of highlights(summary) of same example can be seen"""

cleaned_ai_medical_chatbot_ds["train"][0]

cleaned_ai_medical_chatbot_ds["train"][1]

from huggingface_hub import notebook_login

notebook_login()

from evaluate import load

MODEL_NAME = "meta-llama/Llama-3.2-1B"

import torch
torch.cuda.empty_cache()

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_data(batch):
    inputs = [f"generate response: {query}" for query in batch["Patient"]]
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=64)
    labels = tokenizer(batch["Doctor"], padding="max_length", truncation=True, max_length=64)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_ds = cleaned_ai_medical_chatbot_ds.map(tokenize_data, batched=True)

training_args = TrainingArguments(
    output_dir="llama3_doc_chat_fine_tuned",
    evaluation_strategy="epoch",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    save_total_limit=2,
    fp16=True,
    gradient_accumulation_steps=2,
)

rouge = evaluate.load("rouge")

def compute_metrics(pred):
    predictions, labels = pred.predictions, pred.label_ids
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    return {
        "rouge1": result["rouge1"].mid.fmeasure,
        "rouge2": result["rouge2"].mid.fmeasure,
        "rougeL": result["rougeL"].mid.fmeasure,
        "rougeLSum": result["rougeLsum"].mid.fmeasure,
    }

    torch.cuda.empty_cache()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()
