import torch
import random
import torch.nn.functional as F

def toVariable(example_batch, param):  # 传入一个batch为单位的example
    batch = len(example_batch)
    maxLenth = 0
    for i in range(len(example_batch)):  # 得到每一个batch中 每一句话的最大长度
        if maxLenth < len(example_batch[i].word_indexes):
            maxLenth = len(example_batch[i].word_indexes)
    x = torch.autograd.Variable(torch.LongTensor(batch, maxLenth))  # 转化为batch*maxLenth的向量
    y = torch.autograd.Variable(torch.LongTensor(batch))  # 转化为 1*batch
    for i in range(0, len(example_batch)):
        for j in range(len(example_batch[i].word_indexes)):
            x.data[i][j] = example_batch[i].word_indexes[j]
            for n in range(len(example_batch[i].word_indexes), maxLenth):  # 这几句是处理没有以batch为单位的大小，剩余的example
                x.data[i][n] = param.unknow
        y.data[i] = example_batch[i].label_index[0]
    return x, y


def getMaxIndex(score):  # 获得最大的下标
    labelsize = score.size()[1]
    max = score.data[0][0]
    maxIndex = 0
    for idx in range(labelsize):
        tmp = score.data[0][idx]
        if max < tmp:
            max = tmp
            maxIndex = idx
    return maxIndex


def train(train_iter, dev_iter, test_iter,  model, param):
    print(param.learnRate)
    optimizer = torch.optim.Adam(model.parameters(), lr=param.learnRate)
    steps = 0
    total_num = len(train_iter)
    batch = param.batch
    # num_batch 表示有多少个batch大小的数据集
    if total_num % batch == 0:
        num_batch = total_num // batch
    else:
        num_batch = total_num // batch + 1
    model.train()

    print(model)
    for epoch in range(1, 30):
        print('\n这是第 {} 次迭代....'.format(epoch))
        print('train_num:', total_num)
        correct = 0
        sum = 0
        avg_loss=0.0
        random.shuffle(train_iter)
        for i in range(num_batch):
            batch_list = []
            # print("Running create batch {}".format(i))
            for j in range(i * batch,
                           (i + 1) * batch if (i + 1) * batch < len(train_iter) else len(train_iter)):
                batch_list.append(train_iter[j])
            random.shuffle(batch_list)  # 进行重新洗牌
            feature, target = toVariable(batch_list, param)
            optimizer.zero_grad()  #

            model.zero_grad()

            if param.LSTM_model:
                if feature.size(0) == param.batch:
                    model.hidden = model.init_hidden(param.num_layers, param.batch)
                else:
                    model.hidden = model.init_hidden(param.num_layers,feature.size(0))

            elif param.BiLSTM_model:
                if feature.size(0) == param.batch:
                    model.hidden = model.init_hidden(param.num_layers, param.batch)
                else:
                    model.hidden = model.init_hidden(param.num_layers,feature.size(0))


            logit = model(feature)
            loss = F.cross_entropy(logit, target)  # 目标函数的求导
            avg_loss += loss.data[0]
            # print('loss:',loss.data[0])
            loss.backward()
            optimizer.step()
            steps += 1
            # if steps % self.param.log_interval==0:
            for i in range(len(target)):
                # pre = logit[i]
                # print(logit[i].size())
                # res = target.data[i]
                if (target.data[i] == getMaxIndex(logit[i].view(1, param.label_size))):
                    correct += 1
                sum += 1

        avg_loss = avg_loss/ sum
        accuracy = correct / sum
        print('Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                           accuracy,
                                                                           correct,
                                                                           sum))
        # print('train acc:{} correct / sum {} / {}'.format(correct / sum, correct, sum))
        eval(dev_iter, model, param)
        eval(test_iter, model, param)

def eval(data_iter,model,param):
    total_num = len(data_iter)
    print("eval_num:", total_num)
    batch = param.batch
    if total_num % batch == 0:
        num_batch = total_num // batch
    else:
        num_batch = total_num // batch + 1
    # for x in range(1, 2):
    random.shuffle(data_iter)
    # print('这是第%d次迭代' % x)
    correct = 0
    avg_loss=0.0
    sum = 0
    for i in range(num_batch):
        batch_list = []
        # print("Running create batch {}".format(i))
        for j in range(i * batch,
                       (i + 1) * batch if (i + 1) * batch < len(data_iter) else len(
                           data_iter)):
            batch_list.append(data_iter[j])
        random.shuffle(batch_list)  # 进行重新洗牌
        feature, target = toVariable(batch_list, param)

        if param.LSTM_model:
            if feature.size(0) == param.batch:
                model.hidden = model.init_hidden(param.num_layers, param.batch)
            else:
                model.hidden = model.init_hidden(param.num_layers,feature.size(0))

        elif param.BiLSTM_model:
            if feature.size(0) == param.batch:
                model.hidden = model.init_hidden(param.num_layers, param.batch)
            else:
                model.hidden = model.init_hidden(param.num_layers,feature.size(0))

        # optimizer.zero_grad()
        logit = model(feature)

        loss = F.cross_entropy(logit, target, size_average=False)  # 目标函数的求导
        avg_loss+=loss.data[0]
        # print('loss:',loss.  data[0])
        # loss.backward()
        # optimizer.step()
        for i in range(len(target)):
            if (target.data[i] == getMaxIndex(logit[i].view(1, param.label_size))):
                correct += 1
            sum += 1
    size = len(data_iter)
    avg_loss = avg_loss / size
    accuracy = correct / sum
    # print('eval acc:{} correct / sum {} / {}'.format(correct / sum, correct, sum))

    print('Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       correct,
                                                                       sum))