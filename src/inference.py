import argparse
import pandas as pd

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

from utils import MyDataset

## Inference를 위한 Argument 정의 
def define_argparser():
    p = argparse.ArgumentParser()
    
    p.add_argument('--model_fn', default='bert-base-multilingual-cased')
    p.add_argument('--trained_model', default='./model')
    p.add_argument('--max_length', type=int, default=512)
    p.add_argument('--device', default=0)

    config = p.parse_args()

    return config

## Inference 함수 정의
def main(config, dataset):

    ### Fine Tuning한 모델 가중치 불러오기
    model = AutoModelForSequenceClassification.from_pretrained(config.trained_model, num_labels=4).cuda()
    
    ### Tokenizer 불러오기
    tokenizer = AutoTokenizer.from_pretrained(config.model_fn)
    
    ### Inference를 위한 데이터를 Torch Dataset으로 변환
    pred_dataset = MyDataset(dataset['text'], dataset['label'])
    
    ### DataLoader 정의
    pred_data_loader = DataLoader(
        dataset = pred_dataset,
        batch_size=512,
        num_workers=8,
        pin_memory=True,
        shuffle=False,
        drop_last=False
    )
    
    outs = []
    
    ### Pipeline 정의 
    pipe = pipeline(
        "text-classification",
        model=model, 
        tokenizer=tokenizer, 
        device=0, 
    )
    
    ### Pipeline에 Batch로 데이터를 forward pass 하여 결과 도출
    ### Prograss bar를 활용하기 위해 DataLoader를 활용 함
    for inputs in tqdm(pred_data_loader):
        input_text = inputs['text']
        out = pipe(input_text, max_length=config.max_length, batch_size=512, truncation=True)
        out = list(map(lambda x: x['label'], out))
        outs += out
        
    ### 결과 Label을 Degree로 변환 정의
    label2str = {
        "LABEL_0" : "Absent",
        "LABEL_1" : "Mild",
        "LABEL_2" : "Moderate",
        "LABEL_3" : "Severe",
    }
    
    ### 결과 Label을 Degree Code 변환 정의
    label2int = {
        "LABEL_0" : 0,
        "LABEL_1" : 1,
        "LABEL_2" : 2,
        "LABEL_3" : 3,
    }
    
    ### Inference하는 데이터 셋에 예측값 Column 추가 (Degree, Degree Code)
    dataset = dataset.assign(
        pred = outs,
        pred_class = lambda x: x['pred'].map(label2int),
        pred_class_nm = lambda x: x['pred'].map(label2str),
    ).drop(columns=['pred'])
    
    dataset.to_csv("../result/test_results.csv", index=False)

if __name__ == '__main__':
    config = define_argparser()
    
    testset = pd.read_csv("../dataset/ABD_SONO_2013.csv")
    
    main(config, dataset=testset)