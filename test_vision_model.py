import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import requests
from unsloth_zoo.vision_utils import process_vision_info

def load_trained_model(model_path="outputs_vision"):
    """Load the trained vision model"""
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor

def generate_description(sample, model, processor):
    """Generate product description from image and text"""
    system_message = """You are a helpful assistant that generates concise, SEO-optimized product descriptions for an ecommerce platform, specifically tailored for mobile search. 
    Your descriptions should be engaging, highlight key features, and include relevant keywords for better search visibility."""
    
    user_prompt = """Generate a concise, SEO-optimized product description for the following product:

Product Name: {product}
Category: {category}

Please provide a description that is engaging, highlights key features, and includes relevant keywords for mobile search optimization."""

    # Convert sample into messages
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {"role": "user", "content": [
            {"type": "image", "image": sample["image"]},
            {"type": "text", "text": user_prompt.format(
                product=sample["product_name"], 
                category=sample["category"]
            )}
        ]},
    ]
    
    # Apply chat template
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Process vision information
    image_inputs = process_vision_info(messages)
    
    # Tokenize and process
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    # Move to device
    inputs = inputs.to(model.device)
    
    # Generate output
    stop_token_ids = [
        processor.tokenizer.eos_token_id, 
        processor.tokenizer.convert_tokens_to_ids("<end_of_turn>")
    ]
    
    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=256, 
        top_p=1.0, 
        do_sample=True, 
        temperature=0.8, 
        eos_token_id=stop_token_ids, 
        disable_compile=True
    )
    
    # Decode output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )
    
    return output_text[0]

def test_model():
    """Test the trained model with sample data"""
    print("Loading trained model...")
    model, processor = load_trained_model()
    
    # Test with Marvel action figure from tutorial
    sample = {
        "product_name": "Hasbro Marvel Avengers-Serie Marvel Assemble Titan-Held, Iron Man, 30,5 cm Actionfigur",
        "category": "Toys & Games | Toy Figures & Playsets | Action Figures",
        "image": Image.open(
            requests.get(
                "https://m.media-amazon.com/images/I/81+7Up7IWyL._AC_SY300_SX300_.jpg", 
                stream=True
            ).raw
        ).convert("RGB")
    }
    
    print("Generating description...")
    description = generate_description(sample, model, processor)
    
    print(f"\nProduct: {sample['product_name']}")
    print(f"Category: {sample['category']}")
    print(f"\nGenerated Description:\n{description}")
    
    # Free memory
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    test_model()
