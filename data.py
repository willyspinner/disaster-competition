from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
import torch

from model import model_tokenizer
MAX_SEQ_LEN = 128
PAD_INDEX = model_tokenizer.convert_tokens_to_ids(model_tokenizer.pad_token)
UNK_INDEX = model_tokenizer.convert_tokens_to_ids(model_tokenizer.unk_token)

# Fields

label_field = Field(sequential=False, use_vocab=False, batch_first=True)
text_field = Field(use_vocab=False, tokenize=model_tokenizer.encode, lower=False, include_lengths=False, batch_first=True,
                                      fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)

train_fields = [('id', None), ('keyword', None), ('location', None), ('text', text_field), ('target', label_field)]

test_fields = [('id', None), ('keyword', None), ('location', None), ('text', text_field), ('target', label_field)]
# TabularDataset
train_dataset = TabularDataset(path='train-mini.csv', format='CSV', fields=train_fields, skip_header=True)

# Note that test dataset doesnt have any target fields. 
test_dataset = TabularDataset(path='test.csv', format='CSV', fields=test_fields, skip_header=True)

"""
train, test = TabularDataset.splits(path='.', train='train.csv',
                                           test='test.csv', format='CSV', fields=fields, skip_header=True)
"""

# Iterators
def get_data_iterators(device, batch_size):
    train_iter = BucketIterator(train_dataset, batch_size=batch_size, sort_key=lambda x: len(x.text),
                                                            device=device, train=True, sort=True, shuffle=True, sort_within_batch=True)
    test_iter = Iterator(test_dataset, batch_size=batch_size, device=device, train=False, shuffle=False, sort=False)
    return train_iter, test_iter
