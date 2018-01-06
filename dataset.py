import collections
import random
import re
import numpy as np
import torch
import hyperparameter
random.seed(233)
torch.manual_seed(233)

class Instance:
    def __init__(self):
        """
        m_word:生成一句话的单词list
        m_label:生成一句话对应的一个标签
        """
        self.m_word=[]
        self.m_label=0

class Example:
    def __init__(self):
        """
        word_indexes:一个句子所有单词的下标
        label_index：一个句子的标签
        """
        self.word_indexes=[]
        self.label_index=[]

class LoadDoc:
    def readFile(self, path):
        f = open(path, 'r')
        newList=[]
        count = 0
        for line in f.readlines():
            count += 1
            instance=Instance()
            info=line.strip().split("|||")
            instance.m_word=info[0].split(' ')
            instance.m_label=info[1].strip()
            newList.append(instance)

            if count == -1:
                break
        f.close()
        return newList

    # def clean_str(string):
    #     string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    #     string = re.sub(r"\'s", " \'s", string)
    #     string = re.sub(r"\'ve", " \'ve", string)
    #     string = re.sub(r"n\'t", " n\'t", string)
    #     string = re.sub(r"\'re", " \'re", string)
    #     string = re.sub(r"\'d", " \'d", string)
    #     string = re.sub(r"\'ll", " \'ll", string)
    #     string = re.sub(r",", " , ", string)
    #     string = re.sub(r"!", " ! ", string)
    #     string = re.sub(r"\(", " \( ", string)
    #     string = re.sub(r"\)", " \) ", string)
    #     string = re.sub(r"\?", " \? ", string)
    #     string = re.sub(r"\s{2,}", " ", string)
    #
    #     return string.strip()
class GenerateDic:
    """
    这个类主要是生成字典
    """
    def __init__(self):
        self.v_list=[]
        self.v_dict=collections.OrderedDict()      #固定字典的生成

    def produceDic(self,listName):
        for i in range(len(listName)):
            if listName[i] not in self.v_list:
                self.v_list.append(listName[i])
        self.v_list.append("unknow")
        # print("v_list",self.v_list)
        for j in range(len(self.v_list)):
            self.v_dict[self.v_list[j]]=j
        return self.v_dict


class Classifier:
    def __init__(self):
        self.param=hyperparameter.Hyperparmeter()
        self.lableDic = GenerateDic()
        self.wordDic = GenerateDic()

    def ToList(self,InitList):
        """
        :param InitList: 相当于很多个instance集合，instance是一个句子一个标签的集合
        :return:
        """
        wordList=[]
        labelList=[]
        for i in range(len(InitList)):
            for j in InitList[i].m_word:
                wordList.append(j)
            # print("wordlist:",wordList.__sizeof__())
            labelList.append(InitList[i].m_label)
            # print("label",labelList)
        return wordList,labelList

    def change_Sentence_in_Num(self,dic_List,word_dic,label_dic):
        """
        这个函数就是一个查字典的过程
        :param dic_List: 传入的是一个字典类型的list[(key,value),(key,value),(key,value)...]
        :param word_dic:
        :param label_dic:
        :return:
        """
        example_list_id=[]
        for i in range(len(dic_List)):
            exampleId=Example()
            for j in dic_List[i].m_word:
                # print(dic_List[i].m_word[0])
                if j in word_dic:
                    id=word_dic[j]
                else:
                    id=word_dic["unknow"]
                exampleId.word_indexes.append(id)
            num=label_dic[dic_List[i].m_label]
            exampleId.label_index.append(num)
            example_list_id.append(exampleId)
        return example_list_id

    def load_my_vec(self, path, vocab, freqs, k=None):
        word_vecs = {}
        with open(path, encoding="utf-8") as f:
            count = 0
            lines = f.readlines()[1:]
            for line in lines:
                values = line.split(" ")
                word = values[0]
                count += 1
                if word in vocab:  # whether to judge if in vocab
                    vector = []
                    for count, val in enumerate(values):
                        if count == 0:
                            continue
                        if count <= k:
                            vector.append(float(val))
                    word_vecs[word] = vector
        return word_vecs

    def add_unknow_words_by_uniform(self,word_vec,vocab,k=100):
        list_word2vec=[]
        outvec=0            #不在此向量中
        invec=0             #在词向量中的
        for word in vocab:
            if word not in word_vec:
                outvec+=1
                word_vec[word]=np.random.uniform(-0.25, 0.25, k).round(6).tolist()
                list_word2vec.append(word_vec[word])
            else:
                invec+=1
                list_word2vec.append(word_vec[word])

        return list_word2vec

    def preprocess(self,path1,path2,path3):
        loadDocument = LoadDoc()  # 加载数据集
        dataset_train= loadDocument.readFile(path1)
        dataset_test = loadDocument.readFile(path2)
        dataset_dev = loadDocument.readFile(path3)

        (word_train, label_train) = self.ToList(dataset_train)  # 得到句子集合，标签集合
        words_dict = self.wordDic.produceDic(word_train)
        labels_dict = self.lableDic.produceDic(label_train)
        labels_dict.pop("unknow")  # 去掉unknow这个标签
        word2vec = self.load_my_vec(path=self.param.word_Embedding_path,vocab=words_dict, freqs=None, k=300)

        self.param.pretrained_weight = self.add_unknow_words_by_uniform(word_vec=word2vec,vocab=words_dict, k=300)

        # 把所有的词，标签转化为字典的index
        train_iter = self.change_Sentence_in_Num(dataset_train, words_dict,labels_dict)
        test_iter  = self.change_Sentence_in_Num(dataset_test, words_dict,labels_dict)
        dev_iter = self.change_Sentence_in_Num(dataset_dev, words_dict,labels_dict)

        self.param.unknow = words_dict["unknow"]
        self.param.embed_num=self.param.unknow + 1
        self.param.label_size=len(labels_dict)

        return train_iter,test_iter,dev_iter








