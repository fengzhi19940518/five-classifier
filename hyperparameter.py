class Hyperparmeter:
    def __init__(self):
        self.learnRate = 0.001
        self.epochs = 0
        self.hidden_dim = 0     #隐层给的窗口数目
        self.label_size = 0
        self.embed_num = 0
        self.embed_dim = 300
        self.class_num = 5
        self.batch = 128
        self.unknow = None
        self.word_Embedding = True
        self.word_Embedding_path = "./data/converted_word_Subj.txt"
        self.pretrained_weight = None
        self.dropout = 0.4
        self.dropout_embed =0
        self.LSTM_hidden_dim = 300
        self.num_layers = 1             #单层的lstm
        self.LSTM_model = False
        self.BiLSTM_model=True

        self.train_path="./data/raw.clean.train"
        self.test_path="./data/raw.clean.test"
        self.dev_path="./data/raw.clean.dev"



