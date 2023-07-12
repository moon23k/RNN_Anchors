import torch, random
import torch.nn as nn



class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        self.dropout = nn.Dropout(config.dropout_ratio)
        self.net = nn.RNN(config.emb_dim, 
                          config.hidden_dim,
                          num_layers=config.n_layers,
                          batch_first=True, 
                          dropout=config.dropout_ratio,
                          bidirectional=config.bidirectional)

    def forward(self, x):
        x = self.dropout(self.embedding(x)) 
        _, hiddens = self.net(x)
        return hiddens



class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
    
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        self.dropout = nn.Dropout(config.dropout_ratio)
        self.net = nn.RNN(config.emb_dim,
                          config.hidden_dim,
                          num_layers=config.n_layers, 
                          batch_first=True,
                          dropout=config.dropout_ratio,
                          bidirectional=config.bidirectional)
    
        self.fc_out = nn.Linear(config.hidden_dim * config.direction, 
                                config.vocab_size)
    
    
    def forward(self, x, hiddens):
        x = self.dropout(self.embedding(x.unsqueeze(1)))
        out, hiddens = self.net(x, hiddens)
        out = self.fc_out(out.squeeze(1))
        return out, hiddens



class SeqGenRNN(nn.Module):
    def __init__(self, config):
        super(SeqGenRNN, self).__init__()

        self.device = config.device
        self.vocab_size = config.vocab_size
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size, max_len = trg.shape
        outputs = torch.zeros(max_len, batch_size, self.vocab_size).to(self.device)

        dec_input = trg[:, 0]
        hiddens = self.encoder(src)

        for t in range(1, max_len):
            out, hiddens = self.decoder(dec_input, hiddens)
            outputs[t] = out
            pred = out.argmax(-1)
            teacher_force = random.random() < teacher_forcing_ratio
            dec_input = trg[:, t] if teacher_force else pred

        return outputs.contiguous().permute(1, 0, 2)[:, 1:]