# Variational Recurrent Neural Network

PyTorch implementation of the variational recurrent neural networks (VRNN).
- Chung, Junyoung, Kyle Kastner, Laurent Dinh, Kratarth Goel, Aaron C. Courville, and Yoshua Bengio. [A recurrent latent variable model for sequential data](https://proceedings.neurips.cc/paper/2015/file/b618c3210e934362ac261db280128c22-Paper.pdf). NeurIPS, 2015.

This model is also known as the recurrent state space model (RSSM).
- Hafner, Danijar, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson. [Learning latent dynamics for planning from pixels](http://proceedings.mlr.press/v97/hafner19a/hafner19a-supp.pdf). PMLR, 2019.

A version with discrete latent space and Gumbel Softmax reparametrisation is implemented in branch `discrete`.

See results in the [Weights and Biases repository](https://wandb.ai/gsprd/vrnn).
