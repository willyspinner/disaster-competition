import torch
import csv
from torch.utils.data import random_split, DataLoader, Dataset
from model import model_tokenizer
from preprocess import preprocess_text


class DisasterDataset(Dataset):
    def __init__(self, csvfile, preprocess=True, max_len=128):
        self.rows = [] # tuple of (text, target)
        skip = True
        with open(csvfile) as f:
            # load all text in memory, makes it easier.
            reader = csv.DictReader(f)
            self.rows = [(row['text'], int(row['target'])) for row in reader]
        self.max_len = max_len
        self.preprocess =preprocess
        print("loaded dataset {}".format(csvfile))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        text = self.rows[idx][0]
        if self.preprocess:
            text = preprocess_text(text)
        item = {
            'label': self.rows[idx][1],
            'text': text,
            # these need to have the same dims
            'input': model_tokenizer(text, return_tensors='pt',
                                            padding='max_length', max_length=self.max_len)
        }
        return item

# Iterators

def collate_fn(dataset_items):
    # we need to flatten since the model tokenizer returns in (1,MAX_LEN) - making the default
    # collate return (batch_size, 1, MAX_LEN)

    labels = torch.Tensor([d['label'] for d in dataset_items]).type(torch.LongTensor)
    texts = [d['text'] for d in dataset_items]
    attns = torch.stack([d['input']['attention_mask'].flatten() for d in dataset_items])
    input_ids = torch.stack([d['input']['input_ids'].flatten() for d in dataset_items])
    return { "label": labels, "input": { "attention_mask": attns, "input_ids": input_ids }, 'text': texts }


def get_datasets(csvpath, split=[80, 10, 10], preprocess=True):
    dataset = DisasterDataset(csvpath, preprocess=preprocess)
    if len(split) != 3:
        raise Error("Split lengths must be of size 3")
    scaled_lengths = [int(len(dataset) * (x / sum(split))) for x in split ]
    if sum(scaled_lengths) < len(dataset):
        scaled_lengths[0] += len(dataset) - sum(scaled_lengths)
    train, dev, test = random_split(dataset, lengths=scaled_lengths, generator=torch.Generator().manual_seed(42))
    return train, dev, test

def get_dataloaders(csvpath, batch_size, split=[80, 10, 10], preprocess=True):
    train, dev, test = get_datasets(csvpath, split, preprocess)
    train_loader = DataLoader(train, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test, batch_size=batch_size, collate_fn=collate_fn)
    dev_loader = DataLoader(dev, batch_size=batch_size, collate_fn=collate_fn)
    return train_loader, dev_loader, test_loader
