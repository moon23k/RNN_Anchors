## RNN Sequence to Sequence 

> &nbsp; This repository implements Sequence to Sequence models using three types of recurrent neural networks (RNN, LSTM, GRU), and shares a series of codes for comparing the natural language performance of each model. The goal is to compare and analyze the natural language generation capabilities of the three models and establish a baseline for the minimum performance of other natural language processing deep learning models in the future. Except for the type of RNN cell, all model variables and training parameters were set uniformly. For the evaluation of natural language generation capabilities, datasets for machine translation, conversation generation, and document summarization were selected.

<br><br>

## Background

<p align="center">
  <img src="https://github.com/moon23k/LSTM_Anchors/assets/71929682/051d7ace-b763-46ab-a4bb-9f362563da27" width="730"></img>
</p><br>

**Vanilla Recurrent Neural Network Cell** <br>
> RNN processes sequential data by using the previous state's output as the current state's input. It has a simple and intuitive structure, and performs well on very simple sequences. However, one of RNN's drawbacks is its difficulty in learning "long-term dependencies." As the sequence gets longer, it becomes challenging for previous information to be effectively propagated to the current state.

<br>

**Long Short-Term Memory Cell** <br>
> LSTM was proposed to overcome the limitations of RNN. It tackles the issue of long-term dependencies by employing mechanisms called "gates." These gates control the flow of information, preserving important information and discarding irrelevant details. LSTM consists of an input gate, a forget gate, and an output gate, each responsible for determining the significance of the input, retaining the memory of the previous state, and controlling the output, respectively.

<br>

**Gated Recurrent Unit Cell** <br>
> GRU is another variant of recurrent neural networks designed to reduce the complexity of LSTM. It has a simpler structure than LSTM and consists of two gates: an update gate and a reset gate. The update gate controls how much of the previous state's information should be retained, while the reset gate determines how much of the previous state should be forgotten. GRU is computationally efficient compared to LSTM and generally performs well when dealing with relatively small amounts of data.

<br>

**Sequence to Sequence Architecture** <br>
> The Sequence to Sequence architecture is a model structure that literally takes a sequence as input and returns a sequence as output.
It consists of an Encoder and a Decoder.
The Encoder transforms the input sequence into a Context Vector, and the Decoder, based on the Context Vector and the decoder's input at each time step, produces the output.
Thanks to this model structure, it is possible to handle variable-length sequences even when using RNN.


<br><br>

## Experimental Setup
The default values for experimental variables are set as follows, and each value can be modified by editing the config.yaml file. <br>

|  **Tokenizer Setup**                        |  **Model Setup**                 |  **Training Setup**               |
| :---                                        | :---                             | :---                              |
| **`Tokenizer Type:`** &hairsp; `Word Level` | **`Input Dimension:`** `10,000`  | **`Epochs:`** `5`                 |
| **`Vocab Size:`** &hairsp; `10,000`         | **`Output Dimension:`** `10,000` | **`Batch Size:`** `32`            |
| **`PAD Idx, Token:`** &hairsp; `0`, `[PAD]` | **`Embedding Dimension:`** `256` | **`Learning Rate:`** `5e-4`       |
| **`UNK Idx, Token:`** &hairsp; `1`, `[UNK]` | **`Hidden Dimension:`** `512`    | **`iters_to_accumulate:`** `4`    |
| **`BOS Idx, Token:`** &hairsp; `2`, `[BOS]` | **`Bidirectional:`** `True`      | **`Gradient Clip Max Norm:`** `1` |
| **`EOS Idx, Token:`** &hairsp; `3`, `[EOS]` | **`Drop-out Ratio:`** `0.5`      | **`Apply AMP:`** `True`           |

<br>To shorten the training speed, techiques below are used. <br> 
* **Accumulative Loss Update**, as shown in the table above, accumulative frequency has set 4. <br>
* **Application of AMP**, which enables to convert float32 type vector into float16 type vector.

<br><br>


## Results

| Model | BLEU | Epoch Time | AVG GPU | Max GPU |
| :---: | :---: | :---: | :---: | :---: |
| RNN Model | 2.12 | 2m 10s | 0.15GB | 0.63GB |
| LSTM Model | 8.35 | 2m 25s | 0.17GB | 0.80GB |
| GRU Model | 9.75 | 2m 20s | 0.16GB | 0.75GB |

<br>

### ⚫ Dialogue Generation
| Model | ROUGE | Epoch Time | AVG GPU | Max GPU |
| :---: | :---: | :---: | :---: | :---: |
| RNN Model | 0.10 | 2m 7s | 0.15GB | 0.61GB |
| LSTM Model | 0.37 | 2m 23s | 0.17GB | 0.78GB |
| GRU Model | 2.16 | 2m 17s | 0.16GB | 0.72GB |

<br>

### ⚫ Text Summarization
| Model | ROUGE | Epoch Time | AVG GPU | Max GPU |
| :---: | :---: | :---: | :---: | :---: |
| RNN Model | 0.00 | 4m 22s | 0.29GB | 1.22GB |
| LSTM Model | 2.23 | 7m 43s | 0.36GB | 2.10GB |
| GRU Model | 2.19 | 4m 27s | 0.25GB | 1.30GB |

<br><br>


## How to Use

```
├── ckpt                --this dir saves model checkpoints and training logs
├── config.yaml         --this file is for setting up arguments for model, training, and tokenizer
├── data                --this dir is for saving Training, Validataion and Test Datasets
├── model               --this dir contains files for Deep Learning Model
│   ├── __init__.py
│   └── seq2seq.py
├── module              --this dir contains a series of modules
│   ├── data.py
│   ├── generate.py
│   ├── __init__.py
│   ├── model.py
│   ├── test.py
│   └── train.py
├── README.md
├── run.py              --this file includes codes for actual tasks such as training, testing, and inference to carry out the practical aspects of the work
└── setup.py            --this file contains a series of codes for preprocessing data, training a tokenizer, and saving the dataset

```

> **Clone git repo**
```
git clone https://github.com/moon23k/RNN_Anchors
```
<br>

> **Download and Process Dataset**
```
bash setup.py -task [all, translation, dialogue, summarization]
```
<br>

> **Execute the run file**
```
python3 run.py -task [translation, dialogue, summarization]
               -mode [train, test, inference]
               -model [rnn, lstm, gru]
               -search(optional) [greedy, beam]
```
<br><br>

## Reference
* [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
* [LONG SHORT-TERM MEMORY](https://www.bioinf.jku.at/publications/older/2604.pdf)
* [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](https://arxiv.org/pdf/1412.3555.pdf)

<br>
