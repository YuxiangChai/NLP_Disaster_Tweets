# Disaster Tweet Classification

## Requirements

- Python >= 3.6
- Pytorch (with specific cuda version) ([official website](https://pytorch.org/get-started/locally/))
- Transformers (Download [here](https://github.com/huggingface/transformers))
- tqdm

## BERT

Inside the `bert` folder, run the following command.

```
$ python main.py [--train path] [--test path] [--validation path] [--epoch n]
```

The default value:

- train: ../data/disaster_response_messages_training.csv
- test: ../data/disaster_response_messages_test.csv
- validation: ../data/disaster_response_messages_validation.csv
- epoch: 10