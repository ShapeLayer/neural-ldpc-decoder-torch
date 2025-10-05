# Neural LDPC Decoder Torch Implementation

PyTorch implementation of Neural LDPC Decoder and derived research.

## Getting Started

```sh
# if Rye is installed
rye sync
# else
python -m pip install -r requirements.txt
```

### Usage

Refer unit test code in [`test/`](./test/).

## Implementations
### Neural LDPC Decoder

_Path: [`src/neural_ldpc_decoder`](./src/neural_ldpc_decoder/)_  

Dai, Jincheng, et al. "Learning to decode protograph LDPC codes." IEEE Journal on Selected Areas in Communications 39.7 (2021): 1983-1999.

[arXiv:2102.03828](https://arxiv.org/abs/2102.03828)

### Boosted Neural LDPC Decoder

_Path: [`src/boosted_neural_ldpc_decoder`](./src/boosted_neural_ldpc_decoder/)_  

Kwak, Hee-Youl, et al. "Boosting learning for LDPC codes to improve the error-floor performance." Advances in Neural Information Processing Systems 36 (2023): 22115-22131.

[arXiv:2310.07194](https://arxiv.org/abs/2310.07194)
