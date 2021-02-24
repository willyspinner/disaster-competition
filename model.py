from transformers import RobertaTokenizer, RobertaModel
from sklearn.feature_extraction.text import TfidfVectorizer
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from preprocess import preprocess_text

model_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

class RBTModel(nn.Module):
    def __init__(self, device='cpu'):
        super(RBTModel, self).__init__()
        self.name="roberta-model"
        self.device = device
        self.tokenizer = model_tokenizer
        self.roberta_base = RobertaModel.from_pretrained('roberta-base').to(device)
        self.fc1 = nn.Linear(in_features=768, out_features=768, bias=True).to(device)
        self.dp1 = nn.Dropout(p=0.15, inplace=False).to(device)
        self.fc2 = nn.Linear(in_features=768, out_features=2, bias=True).to(device)
        self.softmax = nn.Softmax(dim=-1).to(device)

    def predict(self, raw_text, preprocess=True, raw_softmax=False):
        text = raw_text
        if preprocess:
            text = preprocess_text(text)
        x = self.tokenizer(text, return_tensors="pt", padding=True)
        input_ids = x['input_ids'].to(self.device)
        attn_mask = x['attention_mask'].to(self.device)
        softmax_output = self.forward(input_ids, attention_mask=attn_mask).to(self.device)
        if raw_softmax:
            return softmax_output
        prediction = softmax_output.argmax(dim=-1)
        return prediction

    def generate_losses(self, labels, loss_criterion, test_text=None, encoded_inputs=None):
        """
        returns loss, model_output
        """
        input_ids = encoded_inputs['input_ids'].to(self.device)
        attn_mask = encoded_inputs['attention_mask'].to(self.device)
        labels = labels.to(self.device)

        labels_one_hot = F.one_hot(labels, num_classes=2).type(torch.float).to(self.device)

        model_output = self.forward(input_ids, attention_mask=attn_mask).to(self.device)
        loss = loss_criterion(model_output, labels_one_hot)

        return loss, model_output


    def forward(self, x, **kwargs):

        # NOTE: RobertaClassificationHead uses the first <s> (start string) token features
        # to feed into the model.
        output = self.roberta_base.forward(x, **kwargs).last_hidden_state
        # output is (batch_Size, ts, 178) where ts = 128
        # so we take the 0th timestep features and put it to a series of fcs.
        # this should be ok, since bert is bidirectional.
        output = output[:, 0, :]
        x = self.fc1(output).to(self.device)
        x = self.dp1(x).to(self.device)
        x = self.fc2(x).to(self.device)
        x = self.softmax(x).to(self.device)
        return x

class TfIdfModel:
    """
    A TF-IDF Model that scores each word in the tweet
    by positive or negative (disaster or non-disaster)
    and decides using the largest score's label.
    """

    def __init__(self, device):
        self.device = device
        """
        len(docs) x len(vocab) matrices:
        """
        self.disaster_score = None
        self.non_disaster_score = None
        self.vocab = set()

    def eval(self):
        pass

    def fit(self, train_dataset):
        disaster_tweets = [d['text'] for d in train_dataset if d['label'] == 1]
        non_disaster_tweets = [d['text'] for d in train_dataset if d['label'] == 0]

        disaster_vectorizer = TfidfVectorizer()
        non_disaster_vectorizer = TfidfVectorizer()
        # get mean tfidf for each word across all documents.
        self.disaster_score = np.mean(disaster_vectorizer.fit_transform(disaster_tweets), axis=0).reshape(-1)
        self.non_disaster_score = np.mean(non_disaster_vectorizer.fit_transform(non_disaster_tweets), axis=0).reshape(-1)
        self.disaster_index = { n : i for i, n in enumerate(disaster_vectorizer.get_feature_names()) }
        self.non_disaster_index = { n : i for i, n in enumerate(non_disaster_vectorizer.get_feature_names()) }

        self.vocab = set(self.disaster_index.keys()).union(set(self.non_disaster_index.keys()))

    def forward(self, input_text):
        if isinstance(input_text, list):
            texts = input_text
        else:
            texts = [input_text]
        results = []
        for text in texts:
            text = preprocess_text(text)
            score_positive = 0
            score_negative = 0
            for token in text.split(' '):
                # compute the ratio to get less influence from stopwords.
                idx = self.disaster_index.get(token, -1)
                non_idx = self.non_disaster_index.get(token, -1)

                if idx == -1 or non_idx == -1:
                    continue
                else:
                    dis_score = self.disaster_score[0, idx]
                    nondis_score = self.non_disaster_score[0, non_idx]

                    score_positive += dis_score / nondis_score
                    score_negative += nondis_score / dis_score

            pred = F.softmax(torch.as_tensor([score_negative, score_positive], dtype=torch.double)).to(self.device)
            results.append(pred)
        if isinstance(input_text, list):
            x = torch.stack(results).to(self.device)
            return x
        else:
            return results[0].to(device)

    def predict(self, text, raw_softmax=False):
        softmax_scores = self.forward(text)
        if raw_softmax:
            return softmax_scores
        predicted = torch.argmax(softmax_scores, dim=-1).to(self.device)
        return predicted

    def generate_losses(self, labels, loss_criterion, test_texts=None, encoded_inputs=None):
        """
        returns loss, model_output
        """

        labels = labels.to(self.device)
        labels_one_hot = F.one_hot(labels, num_classes=2).type(torch.float).to(self.device)
        model_output = self.forward(test_texts)
        loss = loss_criterion(model_output.double(), labels_one_hot.double())

        return loss, model_output


class EnsembleModel:
    """
    for use in evaluation only
    """
    def __init__(self, roberta_path, fit_dataset, device, weights = [0.6, 0.4]):
        self. device = device
        self.roberta_model = RBTModel(device)
        self.roberta_model.load_state_dict(torch.load(roberta_path))
        self.roberta_model.eval()
        self.tfidf_model = TfIdfModel()
        self.tfidf_model.fit(fit_dataset)
        self.weights = weights

    def predict(self, texts, encoded_inputs):
        # TODO: not sure if roberta can predict multiple batches.
        preds = weights[0] * self.roberta_model.predict(texts, raw_softmax=True) + \
                weights[1] * self.tfidf_model.predicT(texts, raw_softmax=True)
        preds /= 2
        # then, we get the argmax.
        return torch.argmax(preds, dim=-1)

