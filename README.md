# ignite-wind

A PyTorch-Ignite compatible implementation of the Wasserstein Inception Distance (WInD) introduced in
https://ieeexplore.ieee.org/document/9053325 inspired by https://github.com/streetakos/wasserstein-distance
for evaluating GANs

WInD extends the assumption in state of the art evaluation measure FID of the data distributions being Gaussians and fits Gaussian Mixture Models on the inception features of real and fake data distribution respectively. More details can be read in the original paper.
