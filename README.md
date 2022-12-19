## LSTM Anchors
> The main purpose of this repo is to measure the performance of the **LSTM Encoder-Decoder Model** in three NLG tasks. 
Each is Neural Machine Translation, Dialogue Generation, Abstractive Text Summarization. The model architecture has implemented by referring to the famous **Sequence to Sequence** paper, and WMT14, Daily-Dialogue, Daily-CNN datasets have used for each task.
Machine translation and Dialogue generation deal with relatively short sequences, but summarization tasks deal with very long sequences. Since it is difficult to properly handle long sentences with only the basic Encoder-Decoder structure, a hierarchical model structure is used for summary tasks.
Except for that, all configurations are the same in the three tasks.

<br>
<br>

## Model desc
> As the name **"Sequence-to-Sequence"** suggests, it is an end-to-end sequence model.
The Architecture consists of Encoder and Decoder. In detail, the Encoder first makes Contetx Vectors from Input Sequences. 
And then the Decoder gets Encoder Outputs and Auto Regressive Values from Target sequences as an Input Values to return Target Sequences.
Before Sequence-to-Sequence Architecture was generally applied to various NLP Tasks, Statistical Models outperformed Neural End-to-End models.
This Architecture has proved its significance by opening Neural end-to-end Model Era.


<br>
<br>

## Configurations
| &emsp; **Vocab Config**                            | &emsp; **Model Config**                 | &emsp; **Training Config**               |
| :---                                               | :---                                    | :---                                     |
| **`Vocab Size:`** &hairsp; `30,000`                | **`Input Dimension:`** `30,000`         | **`Epochs:`** `10`                       |
| **`Vocab Type:`** &hairsp; `BPE`                   | **`Output Dimension:`** `30,000`        | **`Batch Size:`** `32`                   |
| **`PAD Idx, Token:`** &hairsp; `0`, `[PAD]` &emsp; | **`Embedding Dimension:`** `256` &emsp; | **`Learning Rate:`** `5e-4`              |
| **`UNK Idx, Token:`** &hairsp; `1`, `[UNK]`        | **`Hidden Dimension:`** `512`           | **`iters_to_accumulate:`** `4`           |
| **`BOS Idx, Token:`** &hairsp; `2`, `[BOS]`        | **`N Layers:`** `2`                     | **`Gradient Clip Max Norm:`** `1` &emsp; |
| **`EOS Idx, Token:`** &hairsp; `3`, `[EOS]`        | **`Drop-out Ratio:`** `0.5`             | **`Apply AMP:`** `True`                  |

<br>
<br>


## Results
> **Training Results**

<center>
  <img src="https://user-images.githubusercontent.com/71929682/201269096-2cc00b2f-4e8d-4071-945c-f5a3bfbca985.png" width="90%" height="70%">
</center>


</br>

> **Test Results**

</br>
</br>


## How to Use
**First clone git repo in your local env**
```
git clone https://github.com/moon23k/LSTM_Anchors
```

<br>

**Download and Process Dataset via setup.py**
```
bash setup.py -task [all, nmt, dialog, sum]
```

<br>

**Execute the run file on your purpose (search is optional)**
```
python3 run.py -task [nmt, dialog, sum] -mode [train, test, inference] -search [greedy, beam]
```


<br>
<br>

## Reference
* [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
