# Neural Question Generation

### Tom Hosking - MSc Project

## NMT

`./src/nmt.py` gives an implementation of [Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](https://arxiv.org/pdf/1406.1078.pdf) with elements of [NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE](https://arxiv.org/pdf/1409.0473.pdf) - both are well provided for in TensorFlow

Features:
  - [x] Tokenisation of Europarl dataset
  - [x] Stacked RNN encoder/decoder
  - [x] Attention mechanism to learn alignments
  - [ ] Beam search decoder for inference - this requires  modifications to the decoder cell, and so needs some work to ensure parameters are shared (that I haven't done)
  - [ ] Pointer network for OOV
  - [ ] Bidirectional RNN - not yet
  - [ ] A saver...
  - [ ] Shuffle training data
  - [ ] Get a feeling for hyperparams - GRU v LSTM? tanh v lrelu?

## Question Generation

[Machine Comprehension by Text-to-Text Neural
Question Generation](https://arxiv.org/pdf/1705.02012.pdf)

ToDo:
 - [ ] Replicate paper
 - [ ] ???
 - [ ] Profit!
