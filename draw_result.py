import matplotlib.pyplot as plt
import pickle


with open('bart/bart_loss.p', 'rb') as f:
    bart_loss = pickle.load(f)
with open('bert/bert_loss.p', 'rb') as f:
    bert_loss = pickle.load(f)
with open('gpt2/gpt2_loss.p', 'rb') as f:
    gpt2_loss = pickle.load(f)
with open('roberta/roberta_loss.p', 'rb') as f:
    roberta_loss = pickle.load(f)
with open('xlnet/xlnet_loss.p', 'rb') as f:
    xlnet_loss = pickle.load(f)


with open('bart/bart_test_result.p', 'rb') as f:
    bart_test_accuracy, bart_correct_0, bart_wrong_0, bart_correct_1, bart_wrong_1 = pickle.load(f)
with open('bert/bert_test_result.p', 'rb') as f:
    bert_test_accuracy, bert_correct_0, bert_wrong_0, bert_correct_1, bert_wrong_1 = pickle.load(f)
with open('gpt2/gpt2_test_result.p', 'rb') as f:
    gpt2_test_accuracy, gpt2_correct_0, gpt2_wrong_0, gpt2_correct_1, gpt2_wrong_1 = pickle.load(f)
with open('roberta/roberta_test_result.p', 'rb') as f:
    roberta_test_accuracy, roberta_correct_0, roberta_wrong_0, roberta_correct_1, roberta_wrong_1 = pickle.load(f)
with open('xlnet/xlnet_test_result.p', 'rb') as f:
    xlnet_test_accuracy, xlnet_correct_0, xlnet_wrong_0, xlnet_correct_1, xlnet_wrong_1 = pickle.load(f)


with open('bart/bart_val_accuracy.p', 'rb') as f:
    bart_val_accuracy = pickle.load(f)
with open('bert/bert_val_accuracy.p', 'rb') as f:
    bert_val_accuracy = pickle.load(f)
with open('gpt2/gpt2_val_accuracy.p', 'rb') as f:
    gpt2_val_accuracy = pickle.load(f)
with open('roberta/roberta_val_accuracy.p', 'rb') as f:
    roberta_val_accuracy = pickle.load(f)
with open('xlnet/xlnet_val_accuracy.p', 'rb') as f:
    xlnet_val_accuracy = pickle.load(f)



x = range(1,11)

plt.figure(1, figsize=(10, 8))
ax1 = plt.subplot(111)
ax1.plot(x, bart_loss, 'r-', label='BART')
ax1.plot(x, bert_loss, 'b-', label='BERT')
ax1.plot(x, gpt2_loss, 'y-', label='GPT2')
ax1.plot(x, roberta_loss, 'g-', label='RoBERTa')
ax1.plot(x, xlnet_loss, 'k-', label='XLNet')
ax1.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
ax1.legend()
ax1.set_title('Loss VS Epoch')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
plt.savefig('loss_epoch.png')


name = ['BART', 'BERT', 'GPT2', 'RoBERTa', 'XLNet']
test_accuracy = [bart_test_accuracy, bert_test_accuracy, gpt2_test_accuracy, roberta_test_accuracy, xlnet_test_accuracy]
accuracy_1 = [bart_correct_1/(bart_correct_1+bart_wrong_1),
              bert_correct_1/(bert_correct_1+bert_wrong_1),
              gpt2_correct_1/(gpt2_correct_1+gpt2_wrong_1),
              roberta_correct_1/(roberta_correct_1+roberta_wrong_1),
              xlnet_correct_1/(xlnet_correct_1+xlnet_wrong_1)]
accuracy_0 = [bart_correct_0/(bart_correct_0+bart_wrong_0),
              bert_correct_0/(bert_correct_0+bert_wrong_0),
              gpt2_correct_0/(gpt2_correct_0+gpt2_wrong_0),
              roberta_correct_0/(roberta_correct_0+roberta_wrong_0),
              xlnet_correct_0/(xlnet_correct_0+xlnet_wrong_0)]
plt.figure(2, figsize=(10, 8))
ax2 = plt.subplot(111)
ax2.scatter(name, test_accuracy, c='r', s=60, label='Test Accuracy')
ax2.scatter(name, accuracy_1, c='b', s=60, label='Disaster-Related Accuracy')
ax2.scatter(name, accuracy_0, c='g', s=60, label='Non-Disaster-Related Accuracy')
ax2.set_title('Test Accuracy VS Model')
ax2.set_xlabel('Model')
ax2.set_ylabel('Test Accuracy')
ax2.legend()
plt.savefig('test_accuracy.png')


plt.figure(3, figsize=(10, 8))
ax3 = plt.subplot(111)
ax3.plot(x, bart_val_accuracy, 'r-', label='BART')
ax3.plot(x, bert_val_accuracy, 'b-', label='BERT')
ax3.plot(x, gpt2_val_accuracy, 'y-', label='GPT2')
ax3.plot(x, roberta_val_accuracy, 'g-', label='RoBERTa')
ax3.plot(x, xlnet_val_accuracy, 'k-', label='XLNet')
ax3.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
ax3.legend()
ax3.set_title('Validation Accuracy VS Epoch')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Validation Accuracy')
plt.savefig('val_accuracy_epoch.png')
