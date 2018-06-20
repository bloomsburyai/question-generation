# Neural Question Generation

### Tom Hosking - MSc Project

## Question Generation

This repo primarily comprises an implmentation of [Machine Comprehension by Text-to-Text Neural Question Generation](https://arxiv.org/pdf/1705.02012.pdf). It is a work in progress and almost certainly contains bugs!

Requires python 3 and TensorFlow 1.4+

### Usage

To train a model, place `SQuAD` and `Glove` datasets in `./data/` and run `train.sh`. To evaluate a saved model, change the checkpoint path in `src/eval.py` and run `eval.sh`. See `src/flags.py` for a description of available options and hyperparameters.

If you have a saved model checkpoint, you can interact with it using the demo - run `python src/demo/app.py`.

### Code structure

`TFModel` provides a basic starting point and should cover generic boilerplate TF work. `Seq2SeqModel` implements most of the model, including a copy mechanism, encoder/decoder architecture and so on. `MaluubaModel` adds the extra computations required for continued training by policy gradient.

`src/datasources/squad_streamer.py` provides an input pipeline using TensorFlow datasets to do all the preprocessing.

`src/langmodel/lm.py` implements a relatively straightforward LSTM language model.

`src/qa/mpcm.py` implements the [Multi-Perspective Context Matching](https://arxiv.org/pdf/1612.04211.pdf) QA model referenced in the Maluuba paper. NOTE: I have yet to train this successfully beyond 55% F1, there may still be bugs hidden in there.

### ToDo

 - [ ] Train using the RL components
 - [ ] Some config options are still hardcoded (eg restore paths, model type)
 - [ ] The output code is still a bit ad-hoc, and could do with tidying up
 - [ ] `train.py` does a lot of work that would ideally be refactored out
