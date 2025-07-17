import os
os.environ["WANDB_DISABLED"] = "true"
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    MBartForConditionalGeneration, 
    MBart50TokenizerFast,
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
import pandas as pd
import numpy as np
from datasets import Dataset as HFDataset
from sklearn.model_selection import train_test_split
import os
import json
import time
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class BrajHindiDataset:
    def __init__(self, csv_path, test_size=0.1, max_length=128):
        self.csv_path = csv_path
        self.test_size = test_size
        self.max_length = max_length
        self.tokenizer = None
        
    def load_data(self):
        df = pd.read_csv(self.csv_path)
        print(f"Loaded dataset with {len(df)} examples")
        
        braj_texts = df['braj_translation'].tolist()
        hindi_texts = df['hindi_sentence'].tolist()

        train_braj, val_braj, train_hindi, val_hindi = train_test_split(
            braj_texts, hindi_texts, test_size=self.test_size, random_state=42
        )

        return {
            'train_braj': train_braj,
            'train_hindi': train_hindi,
            'val_braj': val_braj,
            'val_hindi': val_hindi
        }
    
    def prepare_datasets(self, tokenizer):
        self.tokenizer = tokenizer
        data = self.load_data()
        
        train_dataset = HFDataset.from_dict({
            'braj': data['train_braj'],
            'hindi': data['train_hindi']
        })

        val_dataset = HFDataset.from_dict({
            'braj': data['val_braj'],
            'hindi': data['val_hindi']
        })
        
        train_dataset = train_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=['hindi', 'braj']
        )
        
        val_dataset = val_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=['hindi', 'braj']
        )
        
        return train_dataset, val_dataset
    
    def preprocess_function(self, examples):
        inputs = [f"Braj: {text}" for text in examples['braj']]
        targets = examples['hindi']
        
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        labels = self.tokenizer(
            targets,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).input_ids
        
        labels[labels == self.tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        
        return model_inputs

class MBartFineTuner:
    def __init__(self, model_name="facebook/mbart-large-50-many-to-many-mmt", output_dir="./fine_tuned_mbart"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        os.makedirs(output_dir, exist_ok=True)
        
    def load_model_and_tokenizer(self):
        print("Loading mBART model and tokenizer...")
        
        self.tokenizer = MBart50TokenizerFast.from_pretrained(self.model_name)
        self.model = MBartForConditionalGeneration.from_pretrained(self.model_name)
        
        self.tokenizer.src_lang = "hi_IN"
        self.tokenizer.tgt_lang = "hi_IN"
        
        special_tokens = ["<braj>", "<hindi>"]
        self.tokenizer.add_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.model.to(self.device)
        
        print(f"Model loaded with {sum(p.numel() for p in self.model.parameters())/1e6:.1f}M parameters")
        print(f"Tokenizer vocabulary size: {len(self.tokenizer)}")
        
        return self.model, self.tokenizer
    
    def setup_training_args(self, num_epochs=3, batch_size=4, learning_rate=5e-5):
        return TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=100,
            eval_steps=500,
            save_steps=1000,
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=True,
            dataloader_num_workers=4,
            remove_unused_columns=False,
            prediction_loss_only=True,
            report_to=None,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            adam_epsilon=1e-8,
            max_grad_norm=1.0,
            save_total_limit=3,
        )
    
    def train(self, train_dataset, val_dataset, training_args):
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            return_tensors="pt"
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        print("Starting fine-tuning...")
        start_time = time.time()
        
        trainer.train()
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time/3600:.2f} hours")
        
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        training_stats = {
            "training_time_hours": training_time/3600,
            "final_eval_loss": trainer.state.log_history[-1].get("eval_loss", "N/A"),
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "model_size_mb": sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024 / 1024,
            "training_date": datetime.now().isoformat()
        }
        
        with open(f"{self.output_dir}/training_stats.json", "w") as f:
            json.dump(training_stats, f, indent=2)
        
        return trainer

class BrajHindiTranslator:
    def __init__(self, model_path="./fine_tuned_mbart"):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        print("Loading fine-tuned model...")
        
        self.tokenizer = MBart50TokenizerFast.from_pretrained(self.model_path)
        self.model = MBartForConditionalGeneration.from_pretrained(self.model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully!")
        
    def translate(self, text, max_length=128, num_beams=5, early_stopping=True):
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        input_text = f"Braj: {text}"
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=early_stopping,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                forced_bos_token_id=self.tokenizer.lang_code_to_id["hi_IN"]
            )

        translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translation.strip()
    
    def batch_translate(self, texts, batch_size=8):
        translations = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Translating"):
            batch_texts = texts[i:i+batch_size]
            batch_translations = []
            
            for text in batch_texts:
                translation = self.translate(text)
                batch_translations.append(translation)
            
            translations.extend(batch_translations)
        
        return translations

def main():
    print("=" * 60)
    print("mBART Fine-tuning for Braj to Hindi Translation")
    print("=" * 60)
    
    dataset_path = "/content/drive/MyDrive/BrajHindiModelV3/Dataset_v1.csv"
    output_dir = "./fine_tuned_mbart_braj_hindi"
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file {dataset_path} not found!")
        return
    
    dataset_processor = BrajHindiDataset(dataset_path, test_size=0.1, max_length=128)
    
    fine_tuner = MBartFineTuner(output_dir=output_dir)
    model, tokenizer = fine_tuner.load_model_and_tokenizer()
    
    print("Preparing datasets...")
    train_dataset, val_dataset = dataset_processor.prepare_datasets(tokenizer)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    training_args = fine_tuner.setup_training_args(
        num_epochs=5,
        batch_size=4,
        learning_rate=3e-5
    )
    
    trainer = fine_tuner.train(train_dataset, val_dataset, training_args)
    
    print("Testing the fine-tuned model...")
    translator = BrajHindiTranslator(output_dir)
    translator.load_model()

    test_sentences = [
        "तोहे देख के मन हरषायो",
        "हमारो घर बड़ो है",
        "कित किताब पढ़त है",
        "म्यां बजार जात हौं",
        "तुम कइसे हो"
    ]

    print("\nTesting translations:")
    print("-" * 40)

    for sentence in test_sentences:
        translation = translator.translate(sentence)
        print(f"Braj: {sentence}")
        print(f"Hindi:  {translation}")
        print("-" * 40)

    print("Fine-tuning completed successfully!")
    print(f"Model saved to: {output_dir}")

if __name__ == "__main__":
    main()
