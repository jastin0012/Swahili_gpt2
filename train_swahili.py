
import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling

# 1. DATA CLEANING
def clean_swahili_data():
    dataset = load_dataset('Adeptschneider/CiviVox-Swahili-text-corpus-v2.0', split='train')
    cleaned = []
    seen = set()
    for example in dataset:
        text = re.sub(r'\s+', ' ', example['text'].strip())
        if len(text.split()) >= 5 and text.lower() not in seen:
            seen.add(text.lower())
            cleaned.append(text)
        if len(cleaned) >= 50000: break
    return cleaned

# 2. TRAINING
def train():
    texts = clean_swahili_data()
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    hf_dataset = Dataset.from_dict({'text': texts})
    tokenized = hf_dataset.map(lambda x: tokenizer(x['text'], truncation=True, max_length=128, padding='max_length'), batched=True)
    
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    args = TrainingArguments(
        output_dir='./swahili-gpt2',
        num_train_epochs=5,
        per_device_train_batch_size=8,
        fp16=True
    )
    
    trainer = Trainer(model=model, args=args, train_dataset=tokenized, data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))
    trainer.train()
    model.save_pretrained('./swahili-gpt2-final')
    tokenizer.save_pretrained('./swahili-gpt2-final')

if __name__ == '__main__':
    train()
