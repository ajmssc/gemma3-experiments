from unsloth import FastModel
import torch
from datasets import load_dataset
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
from trl import SFTTrainer, SFTConfig

# Configuration
max_seq_length = 2048
model_name = "google/gemma-3-4b-pt"  # Vision-capable Gemma model

# Load model with unsloth optimizations
model, tokenizer = FastModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=True,  # Use 4-bit quantization for memory efficiency
    load_in_8bit=False,
    full_finetuning=False,
    # token="hf_...",  # Add your HF token if using gated models
)

# Apply PEFT configuration for vision tasks
model = FastModel.get_peft_model(
    model,
    r=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    lora_alpha=128,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# Load the processor for vision tasks
processor = AutoProcessor.from_pretrained(model_name)

# Load the Amazon product descriptions dataset
dataset = load_dataset("philschmid/amazon-product-descriptions-vlm", split="train[:1000]")

# System and user prompts from the tutorial
system_message = """You are a helpful assistant that generates concise, SEO-optimized product descriptions for an ecommerce platform, specifically tailored for mobile search. 
Your descriptions should be engaging, highlight key features, and include relevant keywords for better search visibility."""

user_prompt = """Generate a concise, SEO-optimized product description for the following product:

Product Name: {product}
Category: {category}

Please provide a description that is engaging, highlights key features, and includes relevant keywords for mobile search optimization."""

def convert_to_vision_format(example):
    """Convert dataset examples to vision format with messages"""
    # Store image separately to avoid Arrow serialization issues
    return {
        "image": example["image"],
        "product_name": example["Product Name"],
        "category": example["Category"],
        "description": example["description"],
        "system_message": system_message,
        "user_prompt": user_prompt.format(
            product=example["Product Name"], 
            category=example["Category"]
        )
    }

# Convert dataset to vision format
dataset = dataset.map(convert_to_vision_format)

def formatting_prompts_func(examples):
    """Format examples for training"""
    texts = []
    for i in range(len(examples["image"])):
        # Create simple text format for vision-text training
        text = f"<image>\n{examples['user_prompt'][i]}\n{examples['description'][i]}"
        texts.append(text)
    return {"text": texts}

# Apply formatting
dataset = dataset.map(formatting_prompts_func, batched=True)

def collate_fn(batch):
    """Custom collate function for vision-text data"""
    texts = [item["text"] for item in batch]
    images = [item["image"] for item in batch]
    
    # Process images using the processor
    image_inputs = processor.image_processor(images, return_tensors="pt")
    
    # Tokenize texts
    text_inputs = processor.tokenizer(
        texts, 
        return_tensors="pt", 
        padding=True,
        truncation=True,
        max_length=max_seq_length
    )
    
    # Combine inputs
    batch_processed = {
        "input_ids": text_inputs["input_ids"],
        "attention_mask": text_inputs["attention_mask"],
        "pixel_values": image_inputs["pixel_values"]
    }
    
    # Create labels and mask padding tokens
    labels = batch_processed["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    
    batch_processed["labels"] = labels
    return batch_processed

# Setup trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    eval_dataset=None,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=2,  # Reduced for vision tasks
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=100,
        learning_rate=5e-6,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs_vision",
        report_to="none",
    ),
    data_collator=collate_fn,
)

# Start training
print("Starting vision fine-tuning...")
trainer.train()

# Save the model
trainer.save_model("outputs_vision")

# Free memory
del model
del trainer
torch.cuda.empty_cache()

print("Vision fine-tuning completed!")
