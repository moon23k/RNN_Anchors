## LSTM Anchors
> The main purpose of this repo is to measure the performance of the **LSTM Encoder-Decoder Model** on three Natural Language Generation tasks. 
Each task is **Neural Machine Translation**, **Dialogue Generation**, and **Abstractive Text Summarization**. The model architecture has implemented by referring to the famous **Sequence to Sequence** paper, and **WMT14 En-De**, **Daily-Dialogue**, and **Daily-CNN** datasets have used for each task.
Machine translation and Dialogue generation deal with relatively short sequences, but Text Summarization covers much longers sequences. Since it is difficult to properly handle long sentences only with the basic Encoder-Decoder structure, a hierarchical model structure has used for summary task.
Except for that, all configurations are the same in the three tasks.

<br>
<br>

## Model desc
> The model architecture here used is LSTM Encoder-Decoder Model. Encoder takes input sequence and convert it to representation vector and Decoder takes encoder output and auto-regressive value to return prediction. But in Summarization task, Encoding process uses two Encoders in a hierarchical way. Each for sentence representation and context representation. 

<br>
<br>

## Experimental Setups
| &emsp; **Vocab Setup**                             | &emsp; **Model Setup**                  | &emsp; **Training Setup**                |
| :---                                               | :---                                    | :---                                     |
| **`Vocab Size:`** &hairsp; `30,000`                | **`Input Dimension:`** `30,000`         | **`Epochs:`** `10`                       |
| **`Tokenizer Type:`** &hairsp; `BPE`               | **`Output Dimension:`** `30,000`        | **`Batch Size:`** `32`                   |
| **`PAD Idx, Token:`** &hairsp; `0`, `[PAD]` &emsp; | **`Embedding Dimension:`** `256` &emsp; | **`Learning Rate:`** `1e-3`              |
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

The table below shows the performance metrics of the three models on test dataset.
**Bleu** and **Rouge** are widely used evaluation metrics for machine translation and summarization, but unfortunately there's no standardized evaluation metric for a dialogue generation task. 
To handle this, arbitrary evaluation metric, named **Similarity Score** has used for Dialogue Generation Task. As the name suggests, this metric measures similarities between label sentences and generated sentences, by using Sentence Bert and cosine distance.

</br>
</br>


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


<br>
<br>

## Reference
* [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
