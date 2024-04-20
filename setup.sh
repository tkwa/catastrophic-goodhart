# git clone --recurse-submodules https://github.com/tkwa/catastrophic-goodhart.git
git submodule init && git submodule update
source activate base
python -m pip install ipykernel pytest torch transformers tqdm transformer_lens wandb matplotlib scipy numpy einops ipywidgets
export PATH="$HOME/.local/bin:$PATH"

sudo apt update

python -m pip install -e ./Open-Assistant/model

code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter