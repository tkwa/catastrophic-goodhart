git clone --recurse-submodules https://github.com/tkwa/catastrophic-goodhart.git
# when already in box, use git submodule init && git submodule update
python -m pip install pytest torch transformers tqdm transformer_lens wandb matplotlib scipy numpy einops
export PATH="$HOME/.local/bin:$PATH"







sudo apt update
sudo apt install wget
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt update

# can't use sudo apt install nvidia-cuda-toolkit; this has the wrong version

# need NVCC/cuda for this
pip install -e ./Open-Assistant/model