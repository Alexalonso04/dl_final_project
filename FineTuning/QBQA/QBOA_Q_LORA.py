
# https://huggingface.co/docs/peft/main/en/task_guides/image_classification_lora
# https://huggingface.co/docs/datasets/en/loading
# https://huggingface.co/docs/bitsandbytes/v0.43.2/fsdp_qlora

from transformers import AutoTokenizer, AutoModelForMultipleChoice, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_int8_training
from datasets import load_dataset
import torch
import evaluate
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import numpy as np

accuracy = evaluate.load("accuracy")

#-----------------------------------------------------------------------------------------------------------------------
# Tokenizer
#-----------------------------------------------------------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"pad_token": "<PAD>"})

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
        labels = [feature.pop(label_name)[0] for feature in features]
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
data_set = "allenai/openbookqa"

df = load_dataset(data_set, "additional", cache_dir="./cache")

train_df, val_df = df['train'], df['validation']

def preprocess_data(examples):
    # https://huggingface.co/docs/transformers/en/tasks/multiple_choice
    first_sentences = [[context] * 4 for context in examples["fact1"]]
    first_sentences = sum(first_sentences, [])

    question_headers = examples["question_stem"]
    second_sentences = []

    for i, header in enumerate(question_headers):
        repeated_header = [header] * 4
        combined_sentences = [f"{h} {end}" for h, end in zip(repeated_header, examples['choices'][i]['text'])]
        second_sentences.extend(combined_sentences)

    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        truncation=True
    )
    answer_key_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    tokenized_examples['label'] = [answer_key_mapping[answer] for answer in examples['answerKey']] * 4
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

# we need to save different model type
model = AutoModelForMultipleChoice.from_pretrained("google-bert/bert-base-uncased",
                                                   quantization_config=bnb_config)

# so GPT2 doesnt have padding?? so need to change embedding size
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = model.config.eos_token_id

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
from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir="checkpoints",
    eval_strategy="steps",
    logging_dir="logs",
    logging_strategy="steps",
    per_device_train_batch_size=8,
    learning_rate=6.0e-5,
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
    #tokenizer=tokenizer,
    processing_class=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

#-----------------------------------------------------------------------------------------------------------------------
# Loop
#-----------------------------------------------------------------------------------------------------------------------
results = trainer.evaluate()
print(results)

results = trainer.train()

# if you have ran before, u should run this one, as it will resume
#results = trainer.train(resume_from_checkpoint=True)
print(results)

results = trainer.evaluate()
print(results)

model.save_pretrained("model.pt")