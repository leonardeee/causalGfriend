# GFRIEND: Causally-Motivated Generative Few-Shot Reward Inference

This is a minimal implementation of the GFRIEND framework from the paper:
"Causally-Motivated Generative Few-Shot Reward Inference through Efficient DPO"

## Usage

```bash
pip install -r requirements.txt
python train.py --data_path data/example_data.json
```
Note: Requires GPU with sufficient VRAM (e.g., A100 40GB+) for Llama-3-8B inference.