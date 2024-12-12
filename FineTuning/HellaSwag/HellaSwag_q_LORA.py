# https://huggingface.co/docs/peft/main/en/task_guides/image_classification_lora
# https://huggingface.co/docs/datasets/en/loading
# https://huggingface.co/docs/bitsandbytes/v0.43.2/fsdp_qlora

from transformers import AutoTokenizer, AutoModelForMultipleChoice, BitsAndBytesConfig, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType,prepare_model_for_int8_training
from datasets import load_dataset
import torch
import evaluate
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import numpy as np
import os
import time

accuracy = evaluate.load("accuracy")
checkpoints_dir = "checkpoints"
data_set = "Rowan/hellaswag"

#-----------------------------------------------------------------------------------------------------------------------
# Tokenizer
#-----------------------------------------------------------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

@dataclass
class DataCollatorForMultipleChoice:
    # https://huggingface.co/docs/transformers/en/tasks/multiple_choice
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [int(feature.pop(label_name)) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}

        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch

#-----------------------------------------------------------------------------------------------------------------------
# Read in dataset
# ARC-C ARC-E BoolQ HellaSwag OBQA PIQA WinoGrande
#-----------------------------------------------------------------------------------------------------------------------
df = load_dataset(data_set, cache_dir="./cache")

train_df, val_df = df['train'], df['validation']

def preprocess_data(examples):
    # https://huggingface.co/docs/transformers/en/tasks/multiple_choice
    first_sentences = [[context] * 4 for context in examples["activity_label"]]
    first_sentences = sum(first_sentences, [])

    question_headers = examples["ctx"]
    second_sentences = []
    for i, header in enumerate(question_headers):
        repeated_header = [header] * 4
        combined_sentences = [f"{h} {end}" for h, end in zip(repeated_header, examples["endings"][i])]
        second_sentences.extend(combined_sentences)

    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    return {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}

encoded_train_df = train_df.map(preprocess_data, batched=True)
encoded_val_df = val_df.map(preprocess_data, batched=True)

encoded_train_df.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
encoded_val_df.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

del train_df
del val_df

#-----------------------------------------------------------------------------------------------------------------------
# Load in model
#-----------------------------------------------------------------------------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,

)

model = AutoModelForMultipleChoice.from_pretrained(
                          "google-bert/bert-base-uncased",
                          num_labels=4,
                          quantization_config=bnb_config)

#model.resize_token_embeddings(len(tokenizer))
#model.config.pad_token_id = model.config.eos_token_id

peft_config = LoraConfig(
                        lora_alpha=16,
                        lora_dropout=0.1,
                        r=8,
                        bias="none",
                        task_type=TaskType.SEQ_CLS,
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

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


#-----------------------------------------------------------------------------------------------------------------------
# Train
#-----------------------------------------------------------------------------------------------------------------------
model.push_to_hub("JTHANGEN/finetuned-hellaswag-gpt2")

args = TrainingArguments(
    output_dir=checkpoints_dir,
    eval_strategy="steps",
    logging_dir="logs",
    logging_strategy="steps",
    save_total_limit=3,
    per_device_train_batch_size=32,
    learning_rate=6.0e-4,
    weight_decay=0.1,
    max_grad_norm=1.0,
    #fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    dataloader_drop_last=True,
    push_to_hub = True,
    hub_model_id = "JTHANGEN/finetuned-hellaswag-gpt2",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=encoded_train_df,
    eval_dataset=encoded_val_df,
    #tokenizer=tokenizer,
    processing_class=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

#-----------------------------------------------------------------------------------------------------------------------
# Prior Fine Tuning
#-----------------------------------------------------------------------------------------------------------------------

eval = trainer.evaluate()
print(f"Evaluation pre Q-LORA: {eval}")

#-----------------------------------------------------------------------------------------------------------------------
# Train
#-----------------------------------------------------------------------------------------------------------------------
continue_prior_run = False

start_time = time.time()

if os.path.isdir(checkpoints_dir) and any(fname.startswith("checkpoint-") for fname in os.listdir(checkpoints_dir)) and continue_prior_run:
  results = trainer.train(resume_from_checkpoint=True)
else:
  results = trainer.train()

print(results)
print(f"Training took {time.time() - start_time} seconds")

#-----------------------------------------------------------------------------------------------------------------------
# After Fine Tuning
#-----------------------------------------------------------------------------------------------------------------------

eval = trainer.evaluate()
print(f"Evaluation post Q-LORA: {eval}")

model.push_to_hub("JTHANGEN/finetuned-hellaswag-gpt2")
tokenizer.push_to_hub("JTHANGEN/finetuned-hellaswag-gpt2")