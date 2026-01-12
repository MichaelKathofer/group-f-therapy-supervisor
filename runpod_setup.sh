#!/bin/bash
# =============================================================================
# RunPod Quick Setup Script for Clarina Supervisor
# =============================================================================
# Run this script after cloning the repo on RunPod:
#   chmod +x runpod_setup.sh && ./runpod_setup.sh
# =============================================================================

echo "=========================================="
echo "CLARINA SUPERVISOR - RUNPOD SETUP"
echo "=========================================="

# Install dependencies
echo "[1/4] Installing Python dependencies..."
pip install -q torch transformers datasets peft trl bitsandbytes accelerate

# Install unsloth (optional, for faster training)
echo "[2/4] Installing Unsloth..."
pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Verify GPU
echo "[3/4] Checking GPU..."
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"

echo "[4/4] Setup complete!"
echo ""
echo "=========================================="
echo "READY TO TRAIN"
echo "=========================================="
echo ""
echo "Run your first iteration (baseline already done):"
echo "  python train_iteration.py --iteration v1.1 --epochs 5 --lr 1e-4"
echo ""
echo "Or run with different LoRA rank:"
echo "  python train_iteration.py --iteration v1.2 --epochs 3 --lora_rank 32"
echo ""
echo "After training, evaluate:"
echo "  python evaluate.py"
echo ""
echo "=========================================="
