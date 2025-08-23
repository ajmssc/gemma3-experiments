from unsloth import FastModel
import torch
from PIL import Image
import requests
from transformers import AutoProcessor
import argparse
import os

def load_trained_model(model_path="outputs_vision"):
    """Load the trained VLM model"""
    print(f"Loading trained model from {model_path}...")
    
    # Load the base model and tokenizer with aggressive memory optimization
    model, tokenizer = FastModel.from_pretrained(
        model_name="google/gemma-3-4b-pt",
        max_seq_length=512,  # Further reduced for memory
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
    )
    
    # Load the trained weights
    model = FastModel.from_pretrained(model_path, model=model)
    
    # Load processor
    processor = AutoProcessor.from_pretrained("google/gemma-3-4b-pt")
    
    return model, tokenizer, processor

def load_image(image_path):
    """Load image from path or URL"""
    if image_path.startswith(('http://', 'https://')):
        image = Image.open(requests.get(image_path, stream=True).raw)
    else:
        image = Image.open(image_path)
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return image

def generate_description(model, tokenizer, processor, image, product_name, category):
    """Generate product description for given image"""
    
    # System and user prompts (same as training)
    system_message = """You are a helpful assistant that generates concise, SEO-optimized product descriptions for an ecommerce platform, specifically tailored for mobile search. 
Your descriptions should be engaging, highlight key features, and include relevant keywords for better search visibility."""

    user_prompt = f"""Generate a concise, SEO-optimized product description for the following product:

Product Name: {product_name}
Category: {category}

Please provide a description that is engaging, highlights key features, and includes relevant keywords for mobile search optimization."""

    # Format the prompt
    text = f"<image>\n{user_prompt}"
    
    # Process inputs
    image_inputs = processor.image_processor([image], return_tensors="pt")
    text_inputs = processor.tokenizer(
        [text], 
        return_tensors="pt", 
        padding=True,
        truncation=True,
        max_length=512  # Further reduced for memory
    )
    
    # Combine inputs
    inputs = {
        "input_ids": text_inputs["input_ids"],
        "attention_mask": text_inputs["attention_mask"],
        "pixel_values": image_inputs["pixel_values"]
    }
    
    # Move to GPU if available, with memory optimization
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        # Clear GPU cache first
        torch.cuda.empty_cache()
        # Use CPU for processing, then move to GPU only for generation
        model = model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate with memory optimization
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,  # Reduced for memory
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part (remove the input prompt)
    input_length = text_inputs["input_ids"].shape[1]
    generated_description = generated_text[input_length:].strip()
    
    return generated_description

def main():
    parser = argparse.ArgumentParser(description="Run inference with trained VLM")
    parser.add_argument("--image", required=True, help="Path or URL to image")
    parser.add_argument("--product_name", required=True, help="Product name")
    parser.add_argument("--category", required=True, help="Product category")
    parser.add_argument("--model_path", default="outputs_vision", help="Path to trained model")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer, processor = load_trained_model(args.model_path)
    
    # Load image
    image = load_image(args.image)
    
    print(f"\nGenerating description for:")
    print(f"Product: {args.product_name}")
    print(f"Category: {args.category}")
    print(f"Image: {args.image}")
    print("-" * 50)
    
    # Generate description
    description = generate_description(
        model, tokenizer, processor, image, 
        args.product_name, args.category
    )
    
    print(f"Generated Description:\n{description}")
    
    # Clean up
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
