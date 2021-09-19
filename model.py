import torch
import torch.nn as nn

from transformers import BertTokenizerFast, BertModel

#tokenizer_bert = BertTokenizerFast.from_pretrained("kykim/bert-kor-base")
#model_bert = BertModel.from_pretrained("kykim/bert-kor-base")

class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()

        # pretrained bert model
        self.bert = BertModel.from_pretrained("kykim/bert-kor-base")
        
        # binary classification
        self.linear = nn.Linear(768, 2)

    def forward(self, data):
        pooler_output = self.bert(**data)[1] # _, pooler_output = self.bert(**data)
        x = self.linear(pooler_output)
        return x