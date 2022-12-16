## LSTM Anchors
> This repo deals with Three Basic Models for Neural Machine Translation Task.
The main purpose is to check the developments while comparing each model.
For a fairer comparision, some modifications are applied and as a result, some parts may differ from those in papers.

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
| &emsp; **Vocab Config**                                | &emsp; **Model Config**                 | &emsp; **Training Config**               |
| :---                                                   | :---                                    | :---                                     |
| **`Vocab Size:`** &hairsp; `30,000`                    | **`Input Dimension:`** `30,000`         | **`Epochs:`** `10`                       |
| **`Vocab Type:`** &hairsp; `Byte Pair Encoding` &emsp; | **`Output Dimension:`** `30,000`        | **`Batch Size:`** `32`                   |
| **`PAD Idx, Token:`** &hairsp; `0`, `[PAD]`            | **`Embedding Dimension:`** `256` &emsp; | **`Learning Rate:`** `5e-4`              |
| **`UNK Idx, Token:`** &hairsp; `1`, `[UNK]`            | **`Hidden Dimension:`** `512`           | **`iters_to_accumulate:`** `4`           |
| **`BOS Idx, Token:`** &hairsp; `2`, `[BOS]`            | **`N Layers:`** `2`                     | **`Gradient Clip Max Norm:`** `1` &emsp; |
| **`EOS Idx, Token:`** &hairsp; `3`, `[EOS]`            | **`Drop-out Ratio:`** `0.5`             | **`Apply AMP:`** `True`                  |

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
