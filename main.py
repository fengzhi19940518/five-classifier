import dataset
import hyperparameter
import torch
import LSTM_model
import BiLSTM_model
import train

classifier=dataset.Classifier()

# train_iter,dev_iter,test_iter=classifier.preprocess("./data/subj.all")
train_iter,dev_iter,test_iter=classifier.preprocess(classifier.param.train_path,classifier.param.test_path,classifier.param.dev_path)
# test_iter=classifier.preprocess(classifier.param.test_path)
# dev_iter=classifier.preprocess(classifier.param.dev_path)
if classifier.param.LSTM_model:
    model = LSTM_model.LSTM_model(classifier.param)
else:
    model = BiLSTM_model.BiLSTM_model(classifier.param)

train.train(train_iter, dev_iter, test_iter, model, classifier.param)


