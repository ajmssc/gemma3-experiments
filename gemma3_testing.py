#!/usr/bin/env python3
"""
Gemma3 Fine-tuning Pipeline
Downloads Alpaca dataset and sets up fine-tuning for Gemma3-2B
"""

import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Gemma3FineTuner:
    def __init__(self, model_name="google/gemma-3-1b-pt", dataset_name="tatsu-lab/alpaca"):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.tokenizer = None
        self.model = None
        self.dataset = None
        
    def download_dataset(self):
        """Download and prepare the Alpaca dataset"""
        logger.info(f"Downloading dataset: {self.dataset_name}")
        
        # Load the Alpaca dataset
        self.dataset = load_dataset(self.dataset_name)
        
        # Filter out empty entries and keep only instruction-following format
        def filter_dataset(example):
            return (example['instruction'] is not None and 
                   example['instruction'].strip() != '' and
                   example['input'] is not None and
                   example['output'] is not None)
        
        self.dataset = self.dataset.filter(filter_dataset)
        logger.info(f"Dataset loaded: {len(self.dataset['train'])} training examples")
        
        return self.dataset
    
    def load_model_and_tokenizer(self):
        """Load Gemma3 model and tokenizer"""
        logger.info(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        logger.info("Model and tokenizer loaded successfully")
        return self.model, self.tokenizer
    
    def format_prompt(self, instruction, input_text, output):
        """Format the prompt in instruction-following format"""
        if input_text.strip():
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        return prompt
    
    def tokenize_function(self, examples):
        """Tokenize the dataset"""
        prompts = []
        for i in range(len(examples['instruction'])):
            prompt = self.format_prompt(
                examples['instruction'][i],
                examples['input'][i],
                examples['output'][i]
            )
            prompts.append(prompt)
        
        # Tokenize with truncation but no padding (data collator will handle padding)
        tokenized = self.tokenizer(
            prompts,
            truncation=True,
            padding=False,
            max_length=512,
            return_tensors=None  # Return lists, not tensors
        )
        
        # Set labels to input_ids for causal language modeling
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    def setup_lora(self, r=16, lora_alpha=32, lora_dropout=0.1):
        """Setup LoRA configuration for efficient fine-tuning"""
        logger.info("Setting up LoRA configuration")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        return self.model
    
    def prepare_dataset(self):
        """Prepare the dataset for training"""
        logger.info("Preparing dataset for training")
        
        # Tokenize the dataset
        tokenized_dataset = self.dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=self.dataset['train'].column_names
        )
        
        return tokenized_dataset
    
    def setup_training(self, output_dir="./gemma3-finetuned", num_epochs=3):
        """Setup training arguments and trainer"""
        logger.info("Setting up training configuration")
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            save_steps=500,
            eval_steps=500,
            save_total_limit=2,
            report_to=None,  # Disable wandb
            remove_unused_columns=False,
        )
        
        # Custom data collator for proper padding
        def collate_fn(batch):
            # Extract input_ids and labels
            input_ids = [item['input_ids'] for item in batch]
            labels = [item['labels'] for item in batch]
            
            # Pad sequences
            max_length = max(len(ids) for ids in input_ids)
            padded_input_ids = []
            padded_labels = []
            attention_masks = []
            
            for ids, lbls in zip(input_ids, labels):
                # Pad with pad_token_id
                padding_length = max_length - len(ids)
                padded_input_ids.append(ids + [self.tokenizer.pad_token_id] * padding_length)
                padded_labels.append(lbls + [-100] * padding_length)  # -100 for padding in labels
                attention_masks.append([1] * len(ids) + [0] * padding_length)
            
            return {
                'input_ids': torch.tensor(padded_input_ids),
                'labels': torch.tensor(padded_labels),
                'attention_mask': torch.tensor(attention_masks)
            }
        
        data_collator = collate_fn
        
        # Split dataset for evaluation
        tokenized_dataset = self.prepare_dataset()
        train_dataset = tokenized_dataset['train'].select(range(int(0.9 * len(tokenized_dataset['train']))))
        eval_dataset = tokenized_dataset['train'].select(range(int(0.9 * len(tokenized_dataset['train'])), len(tokenized_dataset['train'])))
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        return trainer
    
    def train(self, output_dir="./gemma3-finetuned", num_epochs=3):
        """Run the complete fine-tuning pipeline"""
        logger.info("Starting fine-tuning pipeline")
        
        # Download dataset
        self.download_dataset()
        
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # Setup LoRA
        self.setup_lora()
        
        # Setup training
        trainer = self.setup_training(output_dir, num_epochs)
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Training completed! Model saved to {output_dir}")
        
        return trainer

def main():
    """Main function to run the fine-tuning"""
    # Initialize the fine-tuner
    fine_tuner = Gemma3FineTuner()
    
    # Run the complete pipeline
    trainer = fine_tuner.train(
        output_dir="./gemma3-finetuned",
        num_epochs=1  # Start with 1 epoch for testing
    )
    
    print("Fine-tuning completed successfully!")

if __name__ == "__main__":
    main()

