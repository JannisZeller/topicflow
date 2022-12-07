# topicflow

This repo contains my attemt on implementing inference for [Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) a generative probabilistic model by Blei, Ng & Jordan (2003) with tensorflow methods. I use the popular toy model from [Griffiths & Steyvers (2004)](https://www.pnas.org/doi/full/10.1073/pnas.0307752101) to overwatch performance or at least whether the different approaches work at all.

The repo is structured in 3 notebooks:

1. `tfp_monte_carlo.ipynb` presents a [Hamiltonian Markov Chain Monte Carlo](https://www.tensorflow.org/probability/examples/A_Tour_of_TensorFlow_Probability#mcmc) Approach that can be performed directly with the functionalities of the [TensorFlow Probability](https://www.tensorflow.org/probability) package (`tfp`).
2. `tfp_variational_inference.ipynb` presents an (approximate) [Variational Inference](https://colab.research.google.com/github/tensorflow/probability/blob/main/tensorflow_probability/examples/jupyter_notebooks/Variational_Inference_and_Joint_Distributions.ipynb) Approach which can yet again be performed directly within `tfp`.
3. `custom_gibbs_sampling.ipynb` presents a custom approach, that implements both a Gibbs sampler (fully vectorized) and a collapsed Gibbs sampler (partly vectorized) as presented in the [lecture by Philipp Hennig](https://youtu.be/z2q7LhsnWNg) which was the initial reason for this repo.

These notebooks are complemented and source from a couple of source files that add wrappers and the actual functionality of the Gibbs samplers as well as a data generation class.


Please understand that I will not provide full background infromation and mathematical "derivations" for everything, this is just a fun/side-project. Please refer to the resources mentioned above for details. I widely rely on the notation from the already mentioned [lecture](https://youtu.be/z2q7LhsnWNg) and use code like `C_DIdK` as representations for objects like 
$$
    C_{dik}\quad \textsf{where} \quad d\in \{1, \dots, D\}, \ \ i \in \{1, \dots , I_d\} \ \ \textsf{and} \ \ k \in \{1, K \}\, .
$$