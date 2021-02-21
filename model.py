from transformers import RobertaTokenizer, RobertaModel
import torch.nn as nn

model_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

class Model(nn.Module):
    def __init__(self, device='cpu'):
        super(Model, self).__init__()
        self.device = device
        self.tokenizer = model_tokenizer
        self.roberta_base = RobertaModel.from_pretrained('roberta-base').to(device)
        self.fc1 = nn.Linear(in_features=768, out_features=768, bias=True).to(device)
        self.dp1 = nn.Dropout(p=0.15, inplace=False).to(device)
        self.fc2 = nn.Linear(in_features=768, out_features=2, bias=True).to(device)
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

        # NOTE: RobertaClassificationHead uses the first <s> (start string) token features
        # to feed into the model.
        output = self.roberta_base.forward(x, **kwargs).last_hidden_state
        # output is (batch_Size, ts, 178) where ts = 128
        # so we take the 0th timestep features and put it to a series of fcs.
        # this should be ok, since bert is bidirectional.
        output = output[:, 0, :]
        # TODO: is 128 a good padding?
        x = self.fc1(output).to(self.device)
        x = self.dp1(x).to(self.device)
        x = self.fc2(x).to(self.device)
        x = self.softmax(x).to(self.device)

        #TODO: perhaps add some layers here? Customize the layers?
        return x
