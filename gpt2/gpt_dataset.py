import torch
import torch.utils.data as data
from transformers import GPT2Tokenizer
import csv


class GPT2Dataset(data.Dataset):
    def __init__(self, path):
        super(GPT2Dataset, self).__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.path = path
        self.sentences = []
        self.labels = []
        self.tokens = []
        self.rows = []
        self.attention_masks = []
        self._get_info()        # get all information we need
    
    def _get_info(self):
        fields = []
        with open(self.path, 'r') as f:
            rd = csv.reader(f)
            fields = next(rd)
            for row in rd:
                dic = {}
                for i in range(len(row)):
                    dic[fields[i]] = row[i]
                self.rows.append(dic)
        for dic in self.rows:
            self.sentences.append(dic['message'])
            if int(dic['related']) == 2:
                self.labels.append(0)
            if int(dic['related']) == 1:
                self.labels.append(1)
            if int(dic['related']) == 0:
                self.labels.append(0)
            encoding = self.tokenizer.encode_plus(
                dic['message'],
                max_length=100,
                truncation=True,
                add_special_tokens=True,
                return_token_type_ids=False,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt'
            )
            self.tokens.append(encoding['input_ids'][0])
            self.attention_masks.append(encoding['attention_mask'][0])
        self.labels = torch.tensor(self.labels)
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, index):
        return self.tokens[index], self.attention_masks[index], self.labels[index]


# c = GPT2Dataset('/home/chai/Documents/final_project/NLP_Disaster_Tweets/data/disaster_response_messages_training.csv')
# print(c[9819])