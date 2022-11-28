import torch

from torch.utils.data import Dataset

## Data Collator 정의
class DataCollator():

    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __call__(self, samples):
        text = [s['text'] for s in samples]
        label = [s['label'] for s in samples]
        
        ### Text(str) 데이터에 Tokenizer 적용
        input_encoding = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
            max_length=self.max_length # 데이터에 맞는 Max Seq Length 정의
        )
        
        ### Label을 모델 학습을 위한 Type으로 변환 (Encoding)
        encode_label = torch.tensor(list(map(lambda x: int(float(x)), label)))
        
        ### Dictionary Type으로 return 할 정보를 정의
        return_value = {
            'input_ids': input_encoding['input_ids'],
            'attention_mask': input_encoding['attention_mask'],
            'labels': encode_label,
        }
            
        return return_value

## Custom Dataset 정의
class MyDataset(Dataset):
    
    def __init__(self, text, labels):
        self.text = text
        self.labels = labels
        
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, idx):
        text = str(self.text[idx])
        label = str(self.labels[idx])

        return {
            'text': text,
            'label': label,
        }