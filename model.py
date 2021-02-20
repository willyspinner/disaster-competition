from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch.nn as nn

model_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.tokenizer = model_tokenizer
        self.roberta_base = RobertaForSequenceClassification.from_pretrained('roberta-base')
        self.softmax = nn.Softmax()

    def predict_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        labels = torch.tensor([1]).unsqueeze(0)
        outputs = self.roberta_base(**inputs, labels=labels)
        print(labels)
        print(outputs)


    def forward(self, x):
        # returns a SequenceClassifierOutput, so get logits.
        logits = self.roberta_base.forward(x).logits
        return self.softmax(logits)
