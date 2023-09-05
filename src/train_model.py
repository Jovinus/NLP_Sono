# %%
import argparse
import os
import json
import numpy as np
import pandas as pd
import torch

from datasets import load_metric
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer
from transformers import TrainingArguments

from utils import DataCollator, MyDataset
# %%
## Model 학습을 위한 Argument 정의 
def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', default='bert-base-multilingual-cased')
    p.add_argument('--save_path', default='./model/')
    p.add_argument('--train_fn', default='../dataset/ABD_SONO_2000_to_2012.csv')
    p.add_argument('--gradient_accumulation_steps', type=int, default=2)
    p.add_argument('--batch_size_per_device', type=int, default=2**5)
    p.add_argument('--n_epochs', type=int, default=10)
    p.add_argument('--warmup_ratio', type=float, default=.2)
    p.add_argument('--max_length', type=int, default=512)
    p.add_argument('--random_state', default=1004, type=int)

    config = p.parse_args()

    return config

## Model 학습 함수
def train_model(config, train_dataset, valid_dataset, save_path):

    ### Tokenizer 불러오기
    tokenizer = AutoTokenizer.from_pretrained(config.model_fn)
    
    ### Pretrained 모델 가중치 불러오기
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_fn, 
        num_labels=4
    )

    print(
        '|train| =', len(train_dataset),
        '|valid| =', len(valid_dataset),
    )

    total_batch_size = config.batch_size_per_device #* torch.cuda.device_count()
    n_total_iterations = int(len(train_dataset) / total_batch_size * config.n_epochs)
    n_warmup_steps = int(n_total_iterations * config.warmup_ratio)

    print(
        '#total_iters =', n_total_iterations,
        '#warmup_iters =', n_warmup_steps,
    )

    ### Downstream Task 수행을 위한 모델 학습 정의
    training_args = TrainingArguments(
        output_dir=os.path.join(save_path, 'checkpoints'),
        num_train_epochs=config.n_epochs,
        per_device_train_batch_size=config.batch_size_per_device,
        per_device_eval_batch_size=config.batch_size_per_device,
        warmup_steps=n_warmup_steps,
        weight_decay=0.01,
        fp16=False,
        logging_steps=n_total_iterations // 100,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        metric_for_best_model='accuracy',
        greater_is_better=True,
        load_best_model_at_end=True,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )

    ### Hugging Face Trainer 정의
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollator(
            tokenizer,
            config.max_length
        ),
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )

    ### Model 학습 및 저장
    trainer.train()
    trainer.model.save_pretrained(save_path)

    ### 학습 조건 저장
    with open(os.path.join(save_path, 'args.json'), 'w') as f:
        json.dump(config.__dict__, f, indent=2)

## 평가지표 계산 정의 (정확도)
def compute_metrics(eval_pred):
    metric = load_metric("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

## Torch Dataset 정의
def make_dataset(config, data):
    ### 80% train, 20% validation에 활용
    train, val = train_test_split(data, test_size=0.2, stratify=data['label'])

    ### Train Set
    train_dataset = MyDataset(
        train['text'].to_list(),
        train['label'].to_list(),
    )

    ### Validation Set
    valid_dataset = MyDataset(
        val['text'].to_list(),
        val['label'].to_list(),
    )

    return train_dataset, valid_dataset

## Main 함수 정의
def main(config):
    data = pd.read_csv(config.train_fn)

    ### Dataset 불러오기
    train_dataset, valid_dataset = make_dataset(config, data)
    
    ### Model 학습
    train_model(config, train_dataset, valid_dataset, config.save_path)


if __name__ == '__main__':
    config = define_argparser()
    
    main(config)
