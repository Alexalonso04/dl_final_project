from transformers import AutoModelForSeq2SeqLM, Trainer, AutoTokenizer
from peft import get_peft_model, get_peft_config, LoraConfig, TaskType
from datasets import load_dataset

model_name = "bigscience/mt0-small"
tokenizer_name = "bigscience/mt0-small"

dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    use_dora=True,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = get_peft_model(model, peft_config)

trainer = Trainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
model.save_pretrained("peft_model")
