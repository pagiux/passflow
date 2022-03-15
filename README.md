<div align="center">    
 
# PassFlow     

[![Paper](https://img.shields.io/badge/arxiv-2105.06165-brightgreen)](https://arxiv.org/abs/2105.06165) [![Conference](https://img.shields.io/badge/DSN-2022-red)](https://dsn2022.github.io/)  [![License](https://img.shields.io/badge/license-MIT-green)](LICENSE) 

</div>

`PassFlow` exploits the properties of Generative Flows to perform password guessing and shows very promising results, being competetive against GAN-based approaches [1, 2].


### Usage
To get the dataset, run

    curl -L --create-dirs -o data/train.txt https://github.com/d4ichi/PassGAN/releases/download/data/rockyou-train.txt
    curl -L --create-dirs -o data/test.txt https://github.com/d4ichi/PassGAN/releases/download/data/rockyou-test.txt


and then run

    pip install tqdm torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html

to install the needed dependencies. We tested using PyTorch 1.7.1 and CUDA 11.0.

Once a model is trained, you can
    
    python main.py --test <checkpoint>
to test the generation using the Static Sampling, or
    
    python main.py --ds <checkpoint>
to test the generation using our Dynamic Sampling with Penalization, or

    python main.py --gs <checkpoint>
to test the generation using our Dynamic Sampling with Gaussian Smoothing, as described in the paper.


### References

[ 1 ] : https://arxiv.org/abs/1709.00440

[ 2 ] : https://arxiv.org/abs/1910.04232

### License

`PassFlow` was made with ♥ and it is released under the MIT license.
