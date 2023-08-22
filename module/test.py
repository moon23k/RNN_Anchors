import math, torch, evaluate



class Tester:
    def __init__(self, config, model, tokenizer, test_dataloader):
        super(Tester, self).__init__()
        
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = test_dataloader

        self.task = config.task
        self.device = config.device
        self.bos_id = config.bos_id
        self.pad_id = config.pad_id
        self.max_len = config.max_len
        self.vocab_size = config.vocab_size
        self.model_type = config.model_type
        
        self.metric_name = 'BLEU' if self.task == 'nmt' else 'ROUGE'
        self.metric_module = evaluate.load(self.metric_name.lower())
        


    def test(self):
        score = 0.0         
        self.model.eval()

        with torch.no_grad():
            for batch in self.dataloader:

                x = batch['src'].to(self.device)
                y = self.tokenize(batch['trg'])            
        
                pred = self.predict(x)
                score += self.evaluate(pred, y)

        txt = f"TEST Result on {self.task.upper()} with {self.model_type.upper()} model"
        txt += f"\n-- Score: {round(score/len(self.dataloader), 2)}\n"
        print(txt)


    def tokenize(self, batch):
        return [self.tokenizer.decode(x) for x in batch.tolist()]


    def predict(self, x):
        batch_size = x.size(0)
        
        output = torch.LongTensor(self.max_len, batch_size, self.vocab_size)
        output = output.fill_(self.pad_id).to(self.device)
        output[0, :] = self.bos_id

        pred = output[0, :]
        hiddens = self.model.encoder(x)

        for t in range(1, self.max_len):
            out, hiddens = self.model.decoder(pred, hiddens)
            output[t] = out
            pred = out.argmax(-1)

        logit = output.contiguous().permute(1, 0, 2)[:, 1:] 
        
        return output


    def evaluate(self, pred, label):
        #For NMT Evaluation
        if self.task == 'nmt':
            score = self.metric_module.compute(
                predictions=pred, 
                references =[[l] for l in label]
            )['bleu']
        #For Dialg & Sum Evaluation
        else:
            score = self.metric_module.compute(
                predictions=pred, 
                references =[[l] for l in label]
            )['rouge2']

        return score * 100
