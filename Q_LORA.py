# https://huggingface.co/docs/peft/main/en/task_guides/image_classification_lora
# https://huggingface.co/docs/datasets/en/loading
# https://huggingface.co/docs/bitsandbytes/v0.43.2/fsdp_qlora

from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig, AutoTokenizer,DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import torch
import numpy as np
import tiktoken
import evaluate

#-----------------------------------------------------------------------------------------------------------------------
# Read in dataset
# ARC-C ARC-E BoolQ HellaSwag OBQA PIQA WinoGrande
#-----------------------------------------------------------------------------------------------------------------------
data_set = "boolq"

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

train_df = load_dataset(data_set, split="train")
test_df = load_dataset(data_set, split="validation")

# we are going to need to create a preprocess function for every dataset
def preprocess_function(examples):
    return tokenizer(
        examples["question"],
        examples["passage"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )

train_dataset = train_df.map(preprocess_function, batched=True)
test_dataset = test_df.map(preprocess_function, batched=True)

train_dataset = train_dataset.rename_column("answer", "labels")
test_dataset = test_dataset.rename_column("answer", "labels")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
#-----------------------------------------------------------------------------------------------------------------------
# Load in model
#-----------------------------------------------------------------------------------------------------------------------

# cannot do 4bit look into how to do 8bit
quantization_config = BitsAndBytesConfig(load_in_8bit=True)


model = AutoModelForSequenceClassification.from_pretrained("openai-community/gpt2",
                                              device_map = 'auto',
                                              quantization_config=quantization_config,
                                              torch_dtype=torch.bfloat16
                                             )

# so GPT2 doesnt have padding?? so need to change embedding size
model.resize_token_embeddings(len(tokenizer))

peft_config = LoraConfig(
                        lora_alpha=16,
                        lora_dropout=0.1,
                        r=64,
                        bias="none",
                        task_type="SEQ_CLS",
                        target_modules="all-linear",
                        )

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
    output_dir="boolq_output"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

train_results = trainer.train()
