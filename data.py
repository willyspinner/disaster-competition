import torch
import csv
from torch.utils.data import random_split, DataLoader, Dataset
from model import model_tokenizer

class DisasterDataset(Dataset):
    def __init__(self, csvfile, max_len=128):
        self.rows = [] # tuple of (text, target)
        skip = True
        with open(csvfile) as f:
            # load all text in memory, makes it easier.
            reader = csv.DictReader(f)
            self.rows = [(row['text'], int(row['target'])) for row in reader]
        self.max_len = max_len
        print("loaded dataset {}".format(csvfile))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        item = {
            'label': self.rows[idx][1],
            # these need to have the same dims
            'encoded_text': model_tokenizer(self.rows[idx][0], return_tensors='pt',
                                            padding='max_length', max_length=self.max_len)
        }
        return item

# Iterators

def collate_fn(dataset_items):
    # we need to flatten since the model tokenizer returns in (1,MAX_LEN) - making the default
    # collate return (batch_size, 1, MAX_LEN)

    labels = torch.Tensor([d['label'] for d in dataset_items]).type(torch.LongTensor)
    attns = torch.stack([d['encoded_text']['attention_mask'].flatten() for d in dataset_items])
    input_ids = torch.stack([d['encoded_text']['input_ids'].flatten() for d in dataset_items])
    return { "label": labels, "encoded_text": { "attention_mask": attns, "input_ids": input_ids } }




def get_dataloaders(device, batch_size, split_lengths=[80, 10, 10]):
    dataset = DisasterDataset('./train.csv')
    if len(split_lengths) != 3:
        raise Error("Split lengths must be of size 3")
    scaled_lengths = [int(len(dataset) * (x / sum(split_lengths))) for x in split_lengths ]
    if sum(scaled_lengths) < len(dataset):
        scaled_lengths[0] += len(dataset) - sum(scaled_lengths)
    train, dev, test = random_split(dataset, lengths=scaled_lengths, generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test, batch_size=batch_size, collate_fn=collate_fn)
    dev_loader = DataLoader(dev, batch_size=batch_size, collate_fn=collate_fn)
    return train_loader, dev_loader, test_loader
