#!/bin/bash
# Training script for optimized MATD3 configuration
# This script activates the rl conda environment and runs the training

echo "===== Starting Optimized MATD3 Training (V2) ====="
echo "Configuration:"
echo "  - Network: 256 latent dim, [256, 256] hidden layers (+100% capacity)"
echo "  - Population size: 6 agents"
echo "  - Batch size: 512 (maximum stability)"
echo "  - Learning rates: Actor 0.0005, Critic 0.002 (faster learning)"
echo "  - Gamma: 0.995 (long-term focus)"
echo "  - Learn step: 25 (2x frequency)"
echo "  - TAU: 0.003 (very slow target updates)"
echo "  - Policy freq: 3 (better actor-critic balance)"
echo "  - Exploration noise: 0.15 (balanced)"
echo "  - Evolution: every 5,000 steps"
echo "  - Max steps: 2,000,000"
echo "  - GPU: Quadro RTX 5000 (16GB)"
echo "=========================================="
echo ""

# Activate conda environment and run training
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl

# Run the training
python main.py

echo ""
echo "===== Training Complete ====="
echo "Check ./models/MATD3/ for saved model and training plots"
