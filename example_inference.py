#!/usr/bin/env python3
"""
Example usage of the trained VLM for product description generation
"""

import subprocess
import sys
import os

def run_inference_example():
    """Run inference with example data"""
    
    # Example 1: Using a local image (if you have one)
    print("Example 1: Local image inference")
    print("=" * 50)
    
    # You can replace this with an actual image path
    example_cmd = [
        "python", "inference.py",
        "--image", "path/to/your/product/image.jpg",
        "--product_name", "Wireless Bluetooth Headphones",
        "--category", "Electronics"
    ]
    
    print("Command to run:")
    print(" ".join(example_cmd))
    print("\nNote: Replace 'path/to/your/product/image.jpg' with actual image path")
    
    # Example 2: Using an image URL
    print("\nExample 2: URL image inference")
    print("=" * 50)
    
    url_example_cmd = [
        "python", "inference.py",
        "--image", "https://example.com/product-image.jpg",
        "--product_name", "Organic Cotton T-Shirt",
        "--category", "Clothing"
    ]
    
    print("Command to run:")
    print(" ".join(url_example_cmd))
    print("\nNote: Replace URL with actual product image URL")
    
    # Example 3: Interactive mode
    print("\nExample 3: Interactive usage")
    print("=" * 50)
    print("You can also import and use the functions directly:")
    print("""
from inference import load_trained_model, load_image, generate_description

# Load model
model, tokenizer, processor = load_trained_model()

# Load image
image = load_image("path/to/image.jpg")

# Generate description
description = generate_description(
    model, tokenizer, processor, image,
    "Product Name", "Category"
)
print(description)
""")

if __name__ == "__main__":
    run_inference_example()
