from unsloth import FastModel
import torch
from datasets import load_dataset
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
from trl import SFTTrainer, SFTConfig
from unsloth_zoo.vision_utils import process_vision_info
from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch


from unsloth import FastVisionModel

model, processor = FastVisionModel.from_pretrained(
    model_name="lora_model",  # YOUR MODEL YOU USED FOR TRAINING
    load_in_4bit=True,  # Set to False for 16bit LoRA
)
FastVisionModel.for_inference(model)  # Enable for inference!




from datasets import load_dataset
dataset = load_dataset("unsloth/LaTeX_OCR", split = "train")

sample = dataset[1]
image = sample["image"].convert("RGB")
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": sample["text"],
            },
            {
                "type": "image",
            },
        ],
    },
]
input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(
    image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt",
).to("cuda")

from transformers import TextStreamer

text_streamer = TextStreamer(processor.tokenizer, skip_prompt=True)
result = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
                   use_cache=True, temperature = 1.0, top_p = 0.95, top_k = 64)

print(result)

