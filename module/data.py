import json, torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence



class Dataset(torch.utils.data.Dataset):

    def __init__(self, tokenizer, task, split):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = self.load_data(task, split)


    @staticmethod
    def load_data(task, split):
        with open(f"data/{task}/{split}.json", 'r') as f:
            data = json.load(f)
        return data


    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        x = self.tokenizer.encode(self.data[idx]['src']).ids
        y = self.tokenizer.encode(self.data[idx]['trg']).ids
        return torch.LongTensor(x), torch.LongTensor(y)



class Collator(object):

    def __init__(self, pad_id):
        self.pad_id = pad_id


    def __call__(self, batch):
        x_batch, y_batch = zip(*batch)
        
        return {'src': self.pad_batch(x_batch), 
                'trg': self.pad_batch(y_batch)}


    def pad_batch(self, batch):
        return pad_sequence(
            batch, 
            batch_first=True, 
            padding_value=self.pad_id
        )



def load_dataloader(config, tokenizer, split):
    is_train = split == 'train'
    batch_size = config.batch_size if is_train \
                 else config.batch_size // 4
    
    return DataLoader(
        Dataset(tokenizer, config.task, split), 
        batch_size=batch_size, 
        shuffle=True if is_train else False,
        collate_fn=Collator(config.pad_id),
        pin_memory=True,
        num_workers=2
    )