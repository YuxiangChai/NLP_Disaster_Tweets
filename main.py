import torch
from torch.utils.data import random_split, DataLoader
import torch.nn as nn

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import GPT2ForSequenceClassification, BertForSequenceClassification, BartForSequenceClassification
from transformers import RobertaForSequenceClassification, XLNetForSequenceClassification

from tqdm import tqdm
import argparse
import pickle

from gpt2.gpt_dataset import GPT2Dataset
from bert.bert_dataset import BertDataset
from bart.bart_dataset import BartDataset
from roberta.roberta_dataset import RobertaDataset
from xlnet.xlnet_dataset import XLNetDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, loader, criterion, optimizer, scheduler):
    model.to(device)
    model.train()
    epoch_loss = 0
    for i, data in enumerate(tqdm(loader, desc='training.....')):
        input_ids = data[0].to(device)
        attention_mask = data[1].to(device)
        label = data[2].to(device)
        optimizer.zero_grad()
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=label)
        loss, logits = output.loss, output.logits
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        scheduler.step()

    tqdm.write('Loss: {}'.format(epoch_loss))
    return epoch_loss


def val(model, loader):
    model.to(device)
    model.eval()
    correct = 0
    wrong = 0

    with torch.no_grad():
        for i, data in enumerate(tqdm(loader, desc='validating...')):
            input_ids = data[0].to(device)
            attention_mask = data[1].to(device)
            label = data[2].to(device)
            try:
                output = model(input_ids=input_ids, attention_mask=attention_mask, labels=label)
            except:
                print(input_ids)
                raise Exception('error')
            loss, logits = output.loss, output.logits
            _, pred = torch.max(logits, dim=1)
            for j in range(len(pred)):
                if pred[j] == label[j]:
                    correct += 1
                else:
                    wrong += 1
    accuracy = correct / (correct + wrong)
    tqdm.write('Accuracy: {:.4f}'.format(accuracy))
    return accuracy


def test(model, loader):
    model.to(device)
    model.eval()
    correct_0 = 0
    wrong_0 = 0
    correct_1 = 0
    wrong_1 = 0

    with torch.no_grad():
        for i, data in enumerate(tqdm(loader, desc='testing......')):
            input_ids = data[0].to(device)
            attention_mask = data[1].to(device)
            label = data[2].to(device)
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=label)
            loss, logits = output.loss, output.logits
            _, pred = torch.max(logits, dim=1)
            for j in range(len(pred)):
                if label[j] == 1:
                    if pred[j] == label[j]:
                        correct_1 += 1
                    else:
                        wrong_1 += 1
                elif label[j] == 0:
                    if pred[j] == label[j]:
                        correct_0 += 1
                    else:
                        wrong_0 += 1
    accuracy = (correct_1 + correct_0) / (correct_1 + wrong_1 + correct_0 + wrong_0)
    tqdm.write('Accuracy: {:.4f}'.format(accuracy))
    return accuracy, correct_0, wrong_0, correct_1, wrong_1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='data/disaster_response_messages_training.csv')
    parser.add_argument('--test', type=str, default='data/disaster_response_messages_test.csv')
    parser.add_argument('--validation', type=str, default='data/disaster_response_messages_validation.csv')
    parser.add_argument('--epoch', type=str, default='10')
    parser.add_argument('--model', type=str, default='bert', choices=['bert', 'bart', 'gpt2', 'roberta', 'xlnet'])
    args = parser.parse_args()
    
    EPOCH = int(args.epoch)
    model_name = args.model

    # create data loader for training and validation
    if model_name == 'bert':
        train_set = BertDataset(args.train)
        val_set = BertDataset(args.validation)
        test_set = BertDataset(args.test)
    elif model_name == 'bart':
        train_set = BartDataset(args.train)
        val_set = BartDataset(args.validation)
        test_set = BartDataset(args.test)
    elif model_name == 'gpt2':
        train_set = GPT2Dataset(args.train)
        val_set = GPT2Dataset(args.validation)
        test_set = GPT2Dataset(args.test)
    elif model_name == 'roberta':
        train_set = RobertaDataset(args.train)
        val_set = RobertaDataset(args.validation)
        test_set = RobertaDataset(args.test)
    elif model_name == 'xlnet':
        train_set = XLNetDataset(args.train)
        val_set = XLNetDataset(args.validation)
        test_set = XLNetDataset(args.test)
    
    train_loader = DataLoader(train_set, batch_size=20, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=20, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=20, shuffle=False)

    print('Data Loaded.')

    if model_name == 'bert':
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    elif model_name == 'gpt2':
        model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=2)
        model.config.pad_token_id = model.config.eos_token_id
    elif model_name == 'bart':
        model = BartForSequenceClassification.from_pretrained('facebook/bart-base', num_labels=2)
    elif model_name == 'roberta':
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
    elif model_name == 'xlnet':
        model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=2)

    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_loader) * EPOCH
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss()

    print('\nModel: ', model_name, '\tEpochs: ', EPOCH)

    epoch_loss = []
    epoch_val_acc = []

    for epoch in range(EPOCH):
        tqdm.write('Epoch: {}'.format(epoch+1))
        loss = train(model, train_loader, criterion, optimizer, scheduler)
        epoch_loss.append(loss)
        val_acc = val(model, val_loader)
        epoch_val_acc.append(val_acc)
    

    torch.save(model, model_name+'/'+model_name+'_model.pt')
    

    # model = torch.load(model_name+'_model.pt')

    tqdm.write('\nFinal test...')
    test_result = test(model, test_loader)

    with open(model_name+'/'+model_name+'_loss.p', 'wb') as f:
        pickle.dump(epoch_loss, f)
    with open(model_name+'/'+model_name+'_val_accuracy.p', 'wb') as f:
        pickle.dump(epoch_val_acc, f)
    with open(model_name+'/'+model_name+'_test_result.p', 'wb') as f:
        pickle.dump(test_result, f)


if __name__ == '__main__':
    main()