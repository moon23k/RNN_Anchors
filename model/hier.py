import random, torch
import torch.nn as nn



class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.emb_dim = config.emb_dim
        self.hidden_dim = config.hidden_dim
        
        self.dropout = nn.Dropout(config.dropout_ratio)
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        
        self.sequence_rnn = nn.LSTM(config.emb_dim,
                                    config.hidden_dim,
                                    config.n_layers,
                                    batch_first=True,
                                    dropout=config.dropout_ratio)
        
        self.context_rnn = nn.LSTM(config.hidden_dim,
                                   config.hidden_dim,
                                   config.n_layers,
                                   batch_first=True,
                                   dropout=config.dropout_ratio)        

    
    def forward(self, x):
        batch_size, seq_num, seq_len = x.shape
        emb_out = self.embedding(x)
        emb_out = emb_out.view(batch_size * seq_num, seq_len, self.emb_dim)
        
        seq_out, (seq_hidden, seq_cell) = self.sequence_rnn(emb_out)
        seq_out = seq_out[:, -1, :]
        seq_out = seq_out.view(batch_size, seq_num, self.hidden_dim)
        
        _, context_hiddens = self.context_rnn(seq_out)
        return context_hiddens



class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        self.rnn = nn.LSTM(config.emb_dim,
                           config.hidden_dim, 
                           config.n_layers,
                           batch_first=True,
                           dropout=config.dropout_ratio)
        self.fc_out = nn.Linear(config.hidden_dim, config.vocab_size)
        self.dropout = nn.Dropout(config.dropout_ratio)

    
    def forward(self, x, hiddens):
        x = x.unsqueeze(1)
        x = self.dropout(self.embedding(x))

        out, hiddens = self.rnn(x, hiddens)
        out = self.fc_out(out.squeeze(1))
        return out, hiddens



class HierSeq2Seq(nn.Module):
    def __init__(self, config):
        super(HierSeq2Seq, self).__init__()
        self.device = config.device
        self.vocab_size = config.vocab_size
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size, max_len = trg.shape
        outputs = torch.ones(max_len, batch_size, self.vocab_size).to(self.device)

        dec_input = trg[:, 0]
        hiddens = self.encoder(src)

        for idx in range(1, max_len):
            out, hiddens = self.decoder(dec_input, hiddens)
            outputs[idx] = out
            pred = out.argmax(-1)
            teacher_force = random.random() < teacher_forcing_ratio
            dec_input = trg[:, idx] if teacher_force else pred

        outputs = outputs.permute(1, 0, 2)
        return outputs[:, 1:].contiguous()