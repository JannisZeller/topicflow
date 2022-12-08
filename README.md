# topicflow

This repo contains my attemt on implementing inference for [Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) a generative probabilistic model by Blei, Ng & Jordan (2003) with tensorflow methods. I use the popular toy model from [Griffiths & Steyvers (2004)](https://www.pnas.org/doi/full/10.1073/pnas.0307752101) to overwatch performance or at least whether the different approaches work at all.

The repo is structured in 3 notebooks:

1. `tfp_monte_carlo.ipynb` presents a [Hamiltonian Markov Chain Monte Carlo](https://www.tensorflow.org/probability/examples/A_Tour_of_TensorFlow_Probability#mcmc) Approach that can be performed directly with the functionalities of the [TensorFlow Probability](https://www.tensorflow.org/probability) package (`tfp`).
2. `tfp_variational_inference.ipynb` presents an (approximate) [Variational Inference](https://colab.research.google.com/github/tensorflow/probability/blob/main/tensorflow_probability/examples/jupyter_notebooks/Variational_Inference_and_Joint_Distributions.ipynb) Approach which can yet again be performed directly within `tfp`.
3. `custom_gibbs_sampling.ipynb` presents a custom approach, that implements both a Gibbs sampler (fully vectorized) and a collapsed Gibbs sampler (partly vectorized) as presented in the [lecture by Philipp Hennig](https://youtu.be/z2q7LhsnWNg) which was the initial reason for this repo.

These notebooks are complemented and source from a couple of source files that add wrappers and the actual functionality of the Gibbs samplers as well as a data generation class.

## Notation

Please understand that I will not provide full background infromation and mathematical "derivations" for everything, this is just a fun/side-project. Please refer to the resources mentioned above for details. I widely rely on the notation from the already mentioned [lecture](https://youtu.be/z2q7LhsnWNg) and use code like `C_DIdK` as representations for objects like 
$$C_{dik}\quad \textsf{where} \quad d\in \{1, \dots, D\}, \ \ i \in \{1, \dots , I_d\} \ \ \textsf{and} \ \ k \in \{1, K \}\, .$$
The idices used are (typically):
- $K$ is the number of topics 
- $D$ is the number of documents
- $I_d$ is the number of words in document $d$
- $N_{\mathrm{max}}$ is the maximum number of words per doument, i. e. $N_{\mathrm{max}} = \max_d \{I_d\}$
- $T$ is the total number of words, i. e. $T = \sum_d I_d$
- $V$ is the vocabulary size which should be a square of an integer for the visualization purposes of this notebook.

## Quantities for Gibbs Sampling

In the Gibbs sampling implementation the Quantities from the mentioned [lecture](https://youtu.be/z2q7LhsnWNg) are needed, i. e. 


**1. N-Tensor**: One efficiency crtitical step is to vectorize the $n$-Tenosr
$$n_{dkv} =  \{i \, \vert \, w_{di} == v \ \texttt{and} \ c_{idk} ==1\}$$
as much as possible. Due to the fact, that the document lengths are variable there is the choice between looping over the number of documents or padding the documents to a unique length. When padded this can be vectorized as the elementwise product of 2 tensors in $\mathbb{R}^3$.

**2. $C$-Tensor**: $c_{dik}$ is the topic assignment of word $i$ of document $d$ in a one-hot encoded manner (only one entry is 1, all others are 0 along the $k$-dimension). Sampling $C$ is sampling from 
$$p(C\vert \Theta, \Pi, W)=\prod_{d=1}^D \prod_{i=1}^{N} \frac{\prod_{k=1}^K \left(\pi_{dk}\theta_{kw_{di}}\right)^{c_{dik}}}{\sum_{k'=1}^K\left(\pi_{dk'}\theta_{k'w_{di}}\right)}$$ 
which can be identitfied as a categorical distribution.

**3. $\Theta$- and $\Pi$-Tensors**: Sampling $\Theta$ and $\Pi$:
$$\begin{align*}p(\Theta\vert C, W) &= \prod_{k=1}^K \mathcal D(\theta_k; \ \beta_{kv} + n_{\cdot kv}) \\ p(\Pi\vert C, W)    &= \prod_{d=1}^D \mathcal D(\pi_d; \ \alpha_{dk} + n_{dk\cdot}) \, .\end{align*}$$
$\Theta$ and $\Pi$ are neither dependent on the number of words per document nor do they have an $I_d$ dimension. The notation $n_{\cdot k v}$ (for example) represents a summation along the "dotted"-axis, i. e.
$$n_{\cdot k v} = \sum_{d=1}^D n_{dkv}\, .$$