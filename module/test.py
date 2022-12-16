import torch, math, time
import torch.nn as nn
from module.search import Search
from torchtext.data.metrics import bleu_score



class Tester:
    def __init__(self, config, model, test_dataloader, tokenizer):
        super(Tester, self).__init__()
        
        self.model = model
        self.task = config.task
        self.tokenizer = tokenizer
        self.device = config.device
        self.dataloader = test_dataloader
        self.batch_size = config.batch_size        
        self.output_dim = config.output_dim
        self.search = Search(config, self.model, tokenizer)
        self.criterion = nn.CrossEntropyLoss(ignore_index=config.pad_id, label_smoothing=0.1).to(self.device)


    def test(self):
        self.model.eval()
        tot_len = len(self.dataloader)
        tot_loss = 0.0
        
        with torch.no_grad():
            for idx, batch in enumerate(self.dataloader):
                src = batch['src'].to(self.device)
                trg = batch['trg'].to(self.device)

                logit = self.model(src, trg, teacher_forcing_ratio=0.0)
                loss = self.criterion(logit.contiguous().view(-1, self.output_dim), 
                                      trg[:, 1:].contiguous().view(-1)).item()
                tot_loss += loss
            
            tot_loss /= tot_len
        
        print(f'Test Results on {self.task} Task')
        print(f">> Test Loss: {tot_loss:.3f} | Test PPL: {math.exp(tot_loss):.2f}\n")


    def get_bleu_score(self, can, ref):
        return bleu_score([self.tokenizer.Decode(can).split()], 
                          [[self.tokenizer.Decode(ref).split()]]) * 100


    def get_rouge_score(self, can, ref):
        rouge_score = None #TBD
        return rouge_score

        
    def inference_test(self):
        self.model.eval()
        batch = next(iter(self.dataloader))
        input_batch = batch['src'].to(self.device)
        label_batch = batch['trg'].to(self.device)

        inference_dicts = []
        for input_seq, label_seq in zip(input_batch, label_batch):
            temp_dict = dict()
            input_seq = self.tokenizer.decode(input_seq.tolist()) 
            label_seq = self.tokenizer.decode(label_seq.tolist())

            temp_dict['input_seq'] = input_seq
            temp_dict['label_seq'] = label_seq

            temp_dict['greedy_out'] = self.search.greedy_search(input_seq)
            temp_dict['beam_out'] = self.search.beam_search(input_seq)

            if self.task == 'nmt':
                temp_dict['greedy_bleu'] = self.get_bleu_score(temp_dict['greedy_out'], label_seq)
                temp_dict['beam_bleu'] = self.get_bleu_score(temp_dict['beam_out'], label_seq)
            elif self.task == 'sum':
                temp_dict['greedy_rouge'] = self.get_rouge_score(temp_dict['greedy_out'], label_seq)
                temp_dict['beam_rouge'] = self.get_rouge_score(temp_dict['beam_out'], label_seq)                    
            
            inference_dicts.append(temp_dict)
        
        if self.task == 'nmt':
            inference_dicts = sorted(inference_dicts, key=lambda d: d['beam_bleu'])
        elif self.task == 'sum':
            inference_dicts = sorted(inference_dicts, key=lambda d: d['beam_rouge'])
            
        print_dicts = [inference_dicts[0]] + \
                      [inference_dicts[self.batch_size // 2]] + \
                      [inference_dicts[-1]]

        print(f'Inference Test on {self.task} model')
        for d in print_dicts:
            print(f">> Input Sequence: {d['input_seq']}")
            print(f">> Label Sequence: {d['label_seq']}")
            
            print(f">> Greedy Sequence: {d['greedy_out']}")
            print(f">> Beam   Sequence : {d['beam_out']}")
            
            if self.task == 'translate':
                print(f">> Greedy BLEU Score: {d['greedy_bleu']:.2f}")
                print(f">> Beam   BLEU Score : {d['beam_bleu']:.2f}\n")

            elif self.task == 'summary':
                print(f">> Greedy ROUGE Score: {d['greedy_rouge']:.2f}")
                print(f">> Beam   ROUGE Score : {d['beam_rouge']:.2f}\n")