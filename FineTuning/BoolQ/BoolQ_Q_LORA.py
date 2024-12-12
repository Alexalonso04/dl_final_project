# https://huggingface.co/docs/peft/main/en/task_guides/image_classification_lora
# https://huggingface.co/docs/datasets/en/loading
# https://huggingface.co/docs/bitsandbytes/v0.43.2/fsdp_qlora

from transformers import AutoModel, AutoTokenizer,DataCollatorWithPadding, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model,prepare_model_for_int8_training
from datasets import load_dataset
import numpy as np
import evaluate

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"pad_token": "<PAD>"})

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

#-----------------------------------------------------------------------------------------------------------------------
# Read in dataset
# ARC-C ARC-E BoolQ HellaSwag OBQA PIQA WinoGrande
#-----------------------------------------------------------------------------------------------------------------------
data_set = "boolq"

df = load_dataset(data_set,cache_dir="./cache")

train_df, val_df = df['train'], df['validation']

def preprocess_data(examples):
    tokenized = tokenizer(
        examples["question"],
        examples["passage"],
        truncation=True,
        padding=True
    )
    tokenized["labels"] = [1 if label == True else 0 for label in examples["answer"]]
    return tokenized

encoded_train_df = train_df.map(preprocess_data, batched=True)
encoded_val_df = val_df.map(preprocess_data, batched=True)

encoded_train_df.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
encoded_val_df.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

del train_df
del val_df

#-----------------------------------------------------------------------------------------------------------------------
# Load in model
#-----------------------------------------------------------------------------------------------------------------------

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,

)


model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-uncased",
                                                   num_labels=2,
                                                           quantization_config=bnb_config)

model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = model.config.eos_token_id

peft_config = LoraConfig(
                        lora_alpha=16,
                        lora_dropout=0.1,
                        r=8,
                        bias="none",
                        task_type="SEQ_CLS",
                        target_modules=["query", "key", "value"]
                        )

model = prepare_model_for_int8_training(model)

model = get_peft_model(
                model,
                peft_config
)

#-----------------------------------------------------------------------------------------------------------------------
# evaluate
#-----------------------------------------------------------------------------------------------------------------------

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


#-----------------------------------------------------------------------------------------------------------------------
# Train
#-----------------------------------------------------------------------------------------------------------------------
from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir="checkpoints",
    eval_strategy="steps", 
    logging_dir="logs",
    logging_strategy="steps",
    per_device_train_batch_size=16,
    learning_rate=6.0e-4,
    weight_decay=0.1,
    max_grad_norm=1.0,
    #fp16=True,
    dataloader_drop_last=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=encoded_train_df,
    eval_dataset=encoded_val_df,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
results = trainer.evaluate()
print(results)

results = trainer.train()

# if you have ran before, u should run this one, as it will resume
#results = trainer.train(resume_from_checkpoint=True)
print(results)

results = trainer.evaluate()
print(results)

model.save_pretrained("model.pt")
# ~30 mins per run