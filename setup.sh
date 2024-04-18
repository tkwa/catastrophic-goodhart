# git clone --recurse-submodules https://github.com/tkwa/catastrophic-goodhart.git
git submodule init && git submodule update
python -m pip install ipykernel pytest torch transformers tqdm transformer_lens wandb matplotlib scipy numpy einops
export PATH="$HOME/.local/bin:$PATH"

sudo apt update

# can't use sudo apt install nvidia-cuda-toolkit; this has the wrong version

# need NVCC/cuda for this
pip install -e ./Open-Assistant/model