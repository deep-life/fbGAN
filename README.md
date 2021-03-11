# fbGAN
Generative Adversarial Network with Feedback Mechanism for Novel Protein Sequences

For more information, check out: https://deep-life.github.io

![alt text](https://raw.githubusercontent.com/deep-life/fbGAN/main/web.png)

## Usage
### Installation
To get the code clone the repository. The necessary packages can be installed by running the following commands in the same directory.

    git clone https://github.com/deep-life/fbGAN.git
    cd fbGAN
    pip install -r requirements.txt
    
### Run the model

    src/run_fbgan.py --task-type DNA --max-len 243 --desired-features B C E 
    
 The script will save automatically results in a separate folder ```experiments```.
