# Neural Question Generation

### Tom Hosking - MSc Project

## Question Generation

[Machine Comprehension by Text-to-Text Neural
Question Generation](https://arxiv.org/pdf/1705.02012.pdf)

`TFModel` provides a basic starting point and should cover generic boilerplate TF work. `SQuADModel` implements loading and processing of the context/question/answer triples using TF, and makes the results (including looked-up versions and seq lengths) available to the model. So, `QGenMaluuba` should only have to worry about implementing specifics of the model.

ToDo:
 - [ ] Replicate paper
 - [ ] ???
 - [ ] Profit!


 ## NMT

 `./src/sandbox/nmt.py` gives an (incredibly ugly) implementation of [Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](https://arxiv.org/pdf/1406.1078.pdf) with elements of [NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE](https://arxiv.org/pdf/1409.0473.pdf) - both are well provided for in TensorFlow

 Features:
   - [x] Tokenisation of Europarl dataset
   - [x] Stacked RNN encoder/decoder
   - [x] Attention mechanism to learn alignments
   - [ ] Beam search decoder for inference - this requires  modifications to the decoder cell, and so needs some work to ensure parameters are shared (that I haven't done)
   - [ ] Pointer network for OOV - this turns out to be very hard when you've preprocessed the data in numpy and named your variables badly...
   - [ ] Bidirectional RNN - not yet
   - [ ] A saver...
   - [ ] Shuffle training data
   - [ ] Get a feeling for hyperparams - GRU v LSTM? tanh v lrelu?
