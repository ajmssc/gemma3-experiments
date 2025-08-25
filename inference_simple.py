from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from PIL import Image
import requests
import argparse
import os

def load_trained_model(model_path="outputs_vision"):
    """Load the trained VLM model"""
    print(f"Loading trained model from {model_path}...")
    
    # Load processor first
    processor = AutoProcessor.from_pretrained("google/gemma-3-4b-pt")
    
    # Load the model directly from the saved path without quantization
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cpu",  # Load on CPU first to avoid memory issues
    )
    
    # Get tokenizer from processor
    tokenizer = processor.tokenizer
    
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
        max_length=512
    )
    
    # Combine inputs
    inputs = {
        "input_ids": text_inputs["input_ids"],
        "attention_mask": text_inputs["attention_mask"],
        "pixel_values": image_inputs["pixel_values"]
    }
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
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
    # main()
    model, tokenizer, processor = load_trained_model("./gemma3-finetuned/outputs_vision")
    
    # Load image
    image = load_image("https://m.media-amazon.com/images/I/81+7Up7IWyL._AC_SY300_SX300_.jpg")
    
    print(f"\nGenerating description for:")
    print(f"Product: Toy Doll")
    print(f"Category: Toys")
    print("-" * 50)
    
    # Generate description
    description = generate_description(
        model, tokenizer, processor, image, 
        "Toy Doll", "Toys"
    )
    
    print(f"Generated Description:\n{description}")
    
    # Clean up
    del model
    torch.cuda.empty_cache()
