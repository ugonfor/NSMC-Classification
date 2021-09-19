import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

import pandas as pd

class NSMCDataset(Dataset):
    def __init__(self, data_path, max_len) -> None:
        super(NSMCDataset, self).__init__()
        # load dataset from .txt file
        column_names = ["id","document","label"]
        self.data = pd.read_csv(data_path, names = column_names, sep='\t')

        # bert tokenizer
        self.tokenizer = BertTokenizerFast.from_pretrained("kykim/bert-kor-base")
        
        # maximum sentence length
        self.max_len = max_len


    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        document = self.data.iloc[index].document
        #id = self.data.iloc[index].id
        label = torch.zeros(2)
        label[int(self.data.iloc[index].label)] = 1

        encode = self.tokenizer(document, padding='max_length', max_length=self.max_len, truncation=True, return_tensors='pt')

        return {'input_ids': encode['input_ids'][0],
                'token_type_ids': encode['token_type_ids'][0],
                'attention_mask': encode['attention_mask'][0],
                'label': label}
        





        

# for test
if __name__ == "__main__":
    
    tokenizer_bert = BertTokenizerFast.from_pretrained("kykim/bert-kor-base")
    print(tokenizer_bert.encode("안녕하세요? 제 이름은 유효곤입니다. ㅎㅎㅎㅎㅎ"))
    print(tokenizer_bert.tokenize("안녕하세요? 제 이름은 유효곤입니다. ㅎㅎㅎㅎㅎ"))
    
    print(tokenizer_bert.cls_token_id)
    print(tokenizer_bert.pad_token_id)
    print(tokenizer_bert.unk_token_id)
    print(tokenizer_bert.sep_token_id)