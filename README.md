This repo contains all experiments in the paper "Catastrophic Goodhart: regularizing RLHF with KL divergence does not mitigate heavy-tailed reward misspecification".

## How to run experiments


* To plot distributions of reward, run `plots_random_pythia.py`, `plots_random_starling.py`, and `plots_from-llama_starling.py`.
* To run Accelerated Coordinate Gradient (ACG) on Starling 7B, run `starling7bexperiment.py`. Then to create plots, run `plots_acg.py`.