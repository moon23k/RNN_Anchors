## RNN Sequence to Sequence 

&nbsp; The main objective of this repository is to measure and compare the performance of Sequence to Sequence models utilizing RNN-based networks in three Natural Language Generation tasks. The models include **RNN**, **LSTM**, and **GRU**, while the tasks are **Neural Machine Translation**, **Dialogue Generation**, and **Abstractive Text Summarization**. The overall model architecture has implemented by referring to the famous **Sequence to Sequence** paper, and **WMT14 En-De**, **Daily-Dialogue**, and **Daily-CNN** datasets have used for each task.

<br><br>

## RNN Types

<p align="center">
  <img src="https://github.com/moon23k/LSTM_Anchors/assets/71929682/051d7ace-b763-46ab-a4bb-9f362563da27" width="730"></img>
</p><br>

**RNN (Recurrent Neural Network)** <br>
> RNN processes sequential data by using the previous state's output as the current state's input. It has a simple and intuitive structure, and performs well on very simple sequences. However, one of RNN's drawbacks is its difficulty in learning "long-term dependencies." As the sequence gets longer, it becomes challenging for previous information to be effectively propagated to the current state.

<br>

**LSTM (Long Short-Term Memory)** <br>
> LSTM was proposed to overcome the limitations of RNN. It tackles the issue of long-term dependencies by employing mechanisms called "gates." These gates control the flow of information, preserving important information and discarding irrelevant details. LSTM consists of an input gate, a forget gate, and an output gate, each responsible for determining the significance of the input, retaining the memory of the previous state, and controlling the output, respectively.

<br>

**GRU (Gated Recurrent Unit)** <br>
> GRU is another variant of recurrent neural networks designed to reduce the complexity of LSTM. It has a simpler structure than LSTM and consists of two gates: an update gate and a reset gate. The update gate controls how much of the previous state's information should be retained, while the reset gate determines how much of the previous state should be forgotten. GRU is computationally efficient compared to LSTM and generally performs well when dealing with relatively small amounts of data.

<br><br>

## Model Architecture

<br><p align="center">
  <img src="https://github.com/moon23k/LSTM_Anchors/assets/71929682/0fe57c1f-e823-456d-b3e1-4c4bd688f00b" width="700"></img>
</p><br>

> The model architecture here used is LSTM Encoder-Decoder Model. Encoder takes input sequence and convert it to representation vector and Decoder takes encoder output and auto-regressive value to return prediction. But in Summarization task, Encoding process uses two Encoders in a hierarchical way. Each for sentence representation and context representation. 

<br><br>

## Experimental Setups
The default values for experimental variables are set as follows, and each value can be modified by editing the config.yaml file. <br>

| &emsp; **Vocab Setup**                             | &emsp; **Model Setup**                  | &emsp; **Training Setup**                |
| :---                                               | :---                                    | :---                                     |
| **`Tokenizer Type:`** &hairsp; `BPE`               | **`Input Dimension:`** `15,000`         | **`Epochs:`** `10`                       |
| **`Vocab Size:`** &hairsp; `15,000`                | **`Output Dimension:`** `15,000`        | **`Batch Size:`** `128`, `32(Sum)`       |
| **`PAD Idx, Token:`** &hairsp; `0`, `[PAD]` &nbsp; | **`Embedding Dimension:`** `256` &nbsp; | **`Learning Rate:`** `1e-3`              |
| **`UNK Idx, Token:`** &hairsp; `1`, `[UNK]`        | **`Hidden Dimension:`** `512`           | **`iters_to_accumulate:`** `4`           |
| **`BOS Idx, Token:`** &hairsp; `2`, `[BOS]`        | **`N Layers:`** `2`                     | **`Gradient Clip Max Norm:`** `1` &nbsp; |
| **`EOS Idx, Token:`** &hairsp; `3`, `[EOS]`        | **`Drop-out Ratio:`** `0.5`             | **`Apply AMP:`** `True`                  |

<br><br>


## Results

| Metric Score | &emsp; RNN Model &emsp; | &emsp; LSTM Model &emsp; | &emsp; GRU Model &emsp; |
|---|:---:|:---:|:---:|
| &nbsp; Machine Translation &nbsp; | - | - | - |
| &nbsp; Dialogue Generation        | - | - | - |
| &nbsp; Text Summarization         | - | - | - |

<br><br>


## How to Use
> **Clone git repo**
```
git clone https://github.com/moon23k/RNN_Anchors
```
<br>

> **Download and Process Dataset**
```
bash setup.py -task [all, nmt, dialog, sum]
```
<br>

> **Execute the run file**
```
python3 run.py -task [nmt, dialog, sum]
               -mode [train, test, inference]
               -model [rnn, lstm, gru]
               -search(optional) [greedy, beam]
```
<br><br>

## Reference
* [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
* [LONG SHORT-TERM MEMORY](https://www.bioinf.jku.at/publications/older/2604.pdf)
* [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](https://arxiv.org/pdf/1412.3555.pdf)

