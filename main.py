import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from peft import PeftModel, PeftConfig


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_dataset("imdb")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


model = AutoModelForSequenceClassification.from_pretrained("gpt2", num_labels=2)
model.config.pad_token_id = model.config.eos_token_id
model.to(device)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="epoch",
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

pre_trained_eval_results = trainer.evaluate()
print("Pre-trained model evaluation results:", pre_trained_eval_results)

# Define LoRA Config
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    target_modules=["c_attn", "c_proj", "c_fc"]
)

# Create PEFT model
peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()

# Set up the trainer for fine-tuning
peft_training_args = TrainingArguments(
    output_dir="./peft_results",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./peft_logs',
    logging_steps=10,
    eval_strategy="epoch",
    fp16=True,
)

peft_trainer = Trainer(
    model=peft_model,
    args=peft_training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

peft_trainer.train()

peft_model.save_pretrained("./peft_model")

fine_tuned_model = PeftModel.from_pretrained(model, "./peft_model")

eval_trainer = Trainer(
    model=fine_tuned_model,
    args=TrainingArguments(output_dir="./pre_trained_eval_results"),
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

fine_tuned_model_eval_result = eval_trainer.evaluate()
print("Fine-tuned model evaluation results:", fine_tuned_model_eval_result)

# Compare results
print("Pre-trained model accuracy:", pre_trained_eval_results['eval_accuracy'])
print("Fine-tuned model accuracy:", fine_tuned_model_eval_result['eval_accuracy'])
print("Improvement:", fine_tuned_model_eval_result['eval_accuracy'] - pre_trained_eval_results['eval_accuracy'])