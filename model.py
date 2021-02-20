from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch.nn as nn

model_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

class Model(nn.Module):
    def __init__(self, device='cpu'):
        super(Model, self).__init__()
        self.device = device
        self.tokenizer = model_tokenizer
        self.roberta_base = RobertaForSequenceClassification.from_pretrained('roberta-base').to(device)
        self.softmax = nn.Softmax(dim=-1).to(device)

    def load(self, model_path):
        # TODO
        pass

    def predict(self, text):
        x= self.tokenizer(text, return_tensors="pt", padding=True)
        input_ids = x['input_ids'].to(self.device)
        attn_mask = x['attention_mask'].to(self.device)
        softmax_output = self.forward(input_ids, attention_mask=attn_mask).to(self.device)
        prediction = softmax_output.argmax(dim=-1)
        return prediction


    def forward(self, x, **kwargs):
        # returns a SequenceClassifierOutput, so get logits.
        logits = self.roberta_base.forward(x, **kwargs).logits.to(self.device)
        #TODO: perhaps add some layers here? Customize the layers?
        return self.softmax(logits)
