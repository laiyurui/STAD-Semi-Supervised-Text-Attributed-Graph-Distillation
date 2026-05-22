# STAD-Semi-Supervised-Text-Attributed-Graph-Distillation
The repository of Semi-Supervised Text-Attributed Graph Distillation accepted to KDD2026

# STAD
### Linux/macOS
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv --python 3.12
source .venv/bin/activate
# Install base dependencies (CPU-only PyTorch)
uv sync
# Install CUDA-enabled PyTorch (automatically detects CUDA version)
CUDA_FULL=$(nvidia-smi | grep -i "cuda version" | grep -oE "[0-9]+\.[0-9]+" | head -1)
MAJOR=$(echo $CUDA_FULL | cut -d. -f1)
CUDA_VERSION=$([ "$MAJOR" -ge 12 ] && echo "cu121" || echo "cu118")
echo "Detected CUDA $CUDA_FULL, installing PyTorch with $CUDA_VERSION"
uv pip uninstall torch torchvision torchaudio
uv pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$CUDA_VERSION"
# Verify GPU is detected
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), 'CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
# Install torch-scatter and torch-sparse
TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_VER=$(python -c "import torch; v='cu'+torch.version.cuda.replace('.', '') if torch.cuda.is_available() and torch.version.cuda else 'cpu'; print(v)")
uv pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_VER}.html
```
## LLM (STAD only)
The text-synthesis step calls a Qwen API. Set the API key via environment variable `QWEN_KEY`, or in `keys.py`, or pass `--qwen_api_key`. Without it, running STAD will raise a clear error when synthesis is reached.
## Data
Download [minilmdata.zip](https://huggingface.co/datasets/zkchen/tsgfm/resolve/main/minilmdata.zip) and extract so that `dataset/0_splits/{dataset_name}/` and `dataset/{dataset_name}/` exist.
## Method
WSD-guided distillation: dual-path encoders (GCN + MLP) with collaborative self-training, WSD-based graph sketching, and LLM synthesis for condensed node text. See `models/GNN.py` for `SelfTrainingGNN`, `tagcond.py` for the full pipeline.
## Running
### Windows (PowerShell)
```powershell
# Activate virtual environment (use batch file to avoid execution policy issues)
.venv\Scripts\activate
# Run the main script
python main.py --reduction_rate 0.05 --dataset cora --tok_lim 256
```
### Linux/macOS
```bash
source .venv/bin/activate
python main.py --reduction_rate 0.05 --dataset cora --tok_lim 256
```
# Reference 
Dataset: https://github.com/CurryTang/TSGFM
LLM4TAG: https://github.com/WxxShirley/LLMNodeBed 
