import random, torch
import torch.nn as nn
from collections import namedtuple



class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        
        self.embedding = nn.Embedding(
            config.vocab_size, 
            config.emb_dim
        )
        self.dropout = nn.Dropout(config.dropout_ratio)

        if self.model_type == 'rnn':
            self.net = nn.RNN(**config.kwargs)
        elif self.model_type == 'lstm':
            self.net = nn.LSTM(**config.kwargs)
        elif self.model_type == 'gru':
            self.net = nn.GRU(**config.kwargs)


    def forward(self, x):
        x = self.dropout(self.embedding(x)) 
        _, hiddens = self.net(x)
        return hiddens



class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
    
        self.embedding = nn.Embedding(
            config.vocab_size, 
            config.emb_dim
        )

        self.dropout = nn.Dropout(config.dropout_ratio)
        
        if self.model_type == 'rnn':
            self.net = nn.RNN(**config.kwargs)
        elif self.model_type == 'lstm':
            self.net = nn.LSTM(**config.kwargs)
        elif self.model_type == 'gru':
            self.net = nn.GRU(**config.kwargs)
    
        self.fc_out = nn.Linear(
            config.hidden_dim * config.direction, 
            config.vocab_size
        )

    
    
    def forward(self, x, hiddens):
        x = self.dropout(self.embedding(x.unsqueeze(1)))
        out, hiddens = self.net(x, hiddens)
        out = self.fc_out(out.squeeze(1))
        return out, hiddens



class Seq2Seq(nn.Module):
    def __init__(self, config):
        super(Seq2Seq, self).__init__()

        self.device = config.device
        
        self.pad_id = config.pad_id
        self.vocab_size = config.vocab_size
        
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.out = namedtuple('Out', 'logit loss')
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.pad_id, 
            label_smoothing=0.1
        ).to(self.device)

    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size, max_len = trg.shape
        
        outputs = torch.Tensor(max_len, batch_size, self.vocab_size)
        outputs = outputs.fill_(self.pad_id).to(self.device)

        dec_input = trg[:, 0]
        hiddens = self.encoder(src)

        for t in range(1, max_len):
            out, hiddens = self.decoder(dec_input, hiddens)
            outputs[t] = out
            pred = out.argmax(-1)
            teacher_force = random.random() < teacher_forcing_ratio
            dec_input = trg[:, t] if teacher_force else pred

        logit = outputs.contiguous().permute(1, 0, 2)[:, 1:] 
        
        self.out.logit = logit
        self.out.loss = self.criterion(
            logit.contiguous().view(-1, self.vocab_size), 
            trg[:, 1:].contiguous().view(-1)
        )
        
        return self.out 
