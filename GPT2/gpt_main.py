import torch
import argparse
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
from tqdm import tqdm

from gpt_dataset import GPT2Dataset
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import GPT2ForSequenceClassification


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

    tqdm.write('Accuracy: {:.4f}'.format(correct / (correct + wrong)))


def test(model, loader):
    model.to(device)
    model.eval()
    correct = 0
    wrong = 0

    with torch.no_grad():
        for i, data in enumerate(tqdm(loader, desc='testing......')):
            input_ids = data[0].to(device)
            attention_mask = data[1].to(device)
            label = data[2].to(device)
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=label)
            loss, logits = output.loss, output.logits
            _, pred = torch.max(logits, dim=1)
            for j in range(len(pred)):
                if pred[j] == label[j]:
                    correct += 1
                else:
                    wrong += 1

    tqdm.write('Accuracy: {:.4f}'.format(correct / (correct + wrong)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='../data/disaster_response_messages_training.csv')
    parser.add_argument('--test', type=str, default='../data/disaster_response_messages_test.csv')
    parser.add_argument('--validation', type=str, default='../data/disaster_response_messages_validation.csv')
    parser.add_argument('--epoch', type=str, default='1')
    args = parser.parse_args()
    
    EPOCH = int(args.epoch)

    # create data loader for training and validation
    train_set = GPT2Dataset(args.train)
    val_set = GPT2Dataset(args.validation)
    train_loader = DataLoader(train_set, batch_size=20, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=20, shuffle=False)

    print('Train data Loaded.')

    model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=2)
    model.config.pad_token_id = model.config.eos_token_id

    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_loader) * EPOCH
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        tqdm.write('Epoch: {}'.format(epoch+1))
        train(model, train_loader, criterion, optimizer, scheduler)
        val(model, val_loader)

    tqdm.write('\nFinal test...')
    test_set = GPT2Dataset(args.test)
    test_loader = DataLoader(test_set, batch_size=20)
    test(model, test_loader)


if __name__ == '__main__':
    main()