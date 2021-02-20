from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch.nn as nn

model_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

class Model(nn.Module):
    def __init__(self, device):
        super(Model, self).__init__()
        self.device = device
        self.tokenizer = model_tokenizer
        self.roberta_base = RobertaForSequenceClassification.from_pretrained('roberta-base').to(device)
        self.softmax = nn.Softmax().to(device)
        print("TYP", type(self.roberta_base.parameters()))
        #self.register_parameter(name='roberta-base', param=self.roberta_base.parameters())

    def predict_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        labels = torch.tensor([1]).unsqueeze(0)
        outputs = self.roberta_base(**inputs, labels=labels)
        print(labels)
        print(outputs)


    def forward(self, x):
        # returns a SequenceClassifierOutput, so get logits.
        logits = self.roberta_base.forward(x).logits.to(self.device)
        return self.softmax(logits)
