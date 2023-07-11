## LSTM Anchors

<br>

> The main purpose of this repo is to measure the performance of the **LSTM Encoder-Decoder Model** on three Natural Language Generation tasks. 
Each task is **Neural Machine Translation**, **Dialogue Generation**, and **Abstractive Text Summarization**. The model architecture has implemented by referring to the famous **Sequence to Sequence** paper, and **WMT14 En-De**, **Daily-Dialogue**, and **Daily-CNN** datasets have used for each task.
Machine translation and Dialogue generation deal with relatively short sequences, but Text Summarization covers much longers sequences. Since it is difficult to properly handle long sentences only with the basic Encoder-Decoder structure, a hierarchical model structure has used for summary task.
Except for that, all configurations are the same in the three tasks.

<br>
<br>

## Model desc

<br><p align="center">
  <img src="https://github.com/moon23k/LSTM_Anchors/assets/71929682/0fe57c1f-e823-456d-b3e1-4c4bd688f00b"></img>
</p><br>

> The model architecture here used is LSTM Encoder-Decoder Model. Encoder takes input sequence and convert it to representation vector and Decoder takes encoder output and auto-regressive value to return prediction. But in Summarization task, Encoding process uses two Encoders in a hierarchical way. Each for sentence representation and context representation. 

<br><br>

## Experimental Setups
The default values for experimental variables are set as follows, and each value can be modified by editing the config.yaml file. <br>

| &emsp; **Vocab Setup**                             | &emsp; **Model Setup**                  | &emsp; **Training Setup**                |
| :---                                               | :---                                    | :---                                     |
| **`Vocab Size:`** &hairsp; `30,000`                | **`Input Dimension:`** `30,000`         | **`Epochs:`** `10`                       |
| **`Tokenizer Type:`** &hairsp; `BPE`               | **`Output Dimension:`** `30,000`        | **`Batch Size:`** `128`, `32(sum)`       |
| **`PAD Idx, Token:`** &hairsp; `0`, `[PAD]` &emsp; | **`Embedding Dimension:`** `256` &emsp; | **`Learning Rate:`** `1e-3`              |
| **`UNK Idx, Token:`** &hairsp; `1`, `[UNK]`        | **`Hidden Dimension:`** `512`           | **`iters_to_accumulate:`** `4`           |
| **`BOS Idx, Token:`** &hairsp; `2`, `[BOS]`        | **`N Layers:`** `2`                     | **`Gradient Clip Max Norm:`** `1` &emsp; |
| **`EOS Idx, Token:`** &hairsp; `3`, `[EOS]`        | **`Drop-out Ratio:`** `0.5`             | **`Apply AMP:`** `True`                  |

<br>To shorten the training speed, three techiques are used. <br> 
* **Pre Tokenization** <br>
* **Accumulative Loss Update**, as shown in the table above, accumulative frequency has set 4. <br>
* **Application of AMP**, which enables to convert float32 type vector into float16 type vector.

<br><br>



## How to Use
**Clone git repo**
```
git clone https://github.com/moon23k/RNN_Anchors
```

<br>

**Download and Process Dataset**
```
bash setup.py -task [all, nmt, dialog, sum]
```

<br>

**Execute the run file**
```
python3 run.py -task [nmt, dialog, sum] -mode [train, test, inference] -search(optional) [greedy, beam]
```
<br><br>

## Reference
* [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
