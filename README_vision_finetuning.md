# Gemma 3 Vision Fine-tuning with Unsloth

This repository contains scripts to fine-tune Gemma 3 models for vision tasks using Unsloth optimizations, based on the [Google AI tutorial](https://ai.google.dev/gemma/docs/core/huggingface_vision_finetune_qlora).

## Setup

1. **Install dependencies:**
   ```bash
   uv pip install "trl==0.15.2"  # Compatible version
   uv pip install unsloth unsloth-zoo
   ```

2. **Accept Gemma license:**
   - Visit [Hugging Face Gemma page](https://huggingface.co/google/gemma-3-4b-pt)
   - Click "Agree and access repository"
   - Get your Hugging Face token

3. **Set up your token:**
   ```python
   # Add your token to the script
   token="hf_your_token_here"
   ```

## Scripts

### 1. `gemma3_vision_simple.py` (Recommended)
- Simplified version that avoids import issues
- Uses standard vision processing without unsloth_zoo dependencies
- Good for getting started

### 2. `gemma3_vision_unsloth.py` (Advanced)
- Full unsloth integration with vision utilities
- Requires working unsloth_zoo installation
- More optimized but may have dependency issues

### 3. `test_vision_model.py`
- Test script to evaluate trained models
- Generates product descriptions from images

## Usage

1. **Train the model:**
   ```bash
   python gemma3_vision_simple.py
   ```

2. **Test the model:**
   ```bash
   python test_vision_model.py
   ```

## Dataset

Uses the `philschmid/amazon-product-descriptions-vlm` dataset which contains:
- Product images
- Product names and categories
- Target descriptions

## Model Configuration

- **Model:** `google/gemma-3-4b-pt` (vision-capable)
- **Quantization:** 4-bit for memory efficiency
- **LoRA:** Rank 128, alpha 128
- **Training:** 100 steps, batch size 2, gradient accumulation 4

## Output

- Trained model saved to `outputs_vision/`
- LoRA adapters that can be merged with base model
- Compatible with Hugging Face ecosystem

## Requirements

- GPU with 16GB+ VRAM (NVIDIA L4/A100 recommended)
- Python 3.8+
- PyTorch 2.4.0+
- Transformers 4.51.3+

## Troubleshooting

If you encounter import errors:
1. Use `gemma3_vision_simple.py` instead
2. Check trl version compatibility
3. Ensure unsloth is properly installed

## References

- [Google AI Vision Fine-tuning Tutorial](https://ai.google.dev/gemma/docs/core/huggingface_vision_finetune_qlora)
- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [TRL Documentation](https://huggingface.co/docs/trl)
