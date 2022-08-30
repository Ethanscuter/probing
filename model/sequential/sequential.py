import collections
import re
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
import pandas as pd

# Read corpus and token text
def read_time_machine():
    """将数据集加载到文本行的列表中"""
    with open('/raid/xwang/corpus/subtitle_tokens.csv', 'r', encoding='utf-8') as f:
    # with open('/home/xwang/project/ACL10WP/toy/data/wiki_toy.csv', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [line.strip().lower() for line in lines]

def tokenize(lines, token='word'):
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)

# Construct vocab
class Vocab:
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

def count_corpus(tokens):
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

def load_corpus_time_machine(max_tokens=-1):
    """返回数据集的词元索引列表和词表"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # 因为数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

# Construct data loader
class SeqDataLoader:
    """加载序列数据的迭代器"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        """Defined in :numref:`sec_language_model`"""
        if use_random_iter:
            self.data_iter_fn = d2l.seq_data_iter_random
        else:
            self.data_iter_fn = d2l.seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

def load_data_time_machine(batch_size, num_steps,
                           use_random_iter=False, max_tokens=10000):
    """返回数据集的迭代器和词表
    Defined in :numref:`sec_language_model`"""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab

# Construct seqential models
class RNNModel(nn.Module):
    """循环神经网络模型"""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # 如果RNN是双向的，num_directions应该是2，否则应该是1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # 全连接层首先将Y的形状改为(时间步数*批量大小,隐藏单元数)
        # 它的输出形状是(时间步数*批量大小,词表大小)。
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量作为隐状态
            return  torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device)
        else:
            # nn.LSTM以元组作为隐状态
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))

def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """训练模型（定义见第8章）"""
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: d2l.predict_ch8(prefix, 50, net, vocab, device)
    # 训练和预测
    for epoch in range(num_epochs):
        ppl, speed = d2l.train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
            print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
            plt.savefig("/home/xwang/project/ACL10WP/plot/subtitle_perplexity.png", dpi=300, bbox_inches="tight")
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))

# predict prob
def predict_probe(prefix, num_preds, net, vocab, device):
    """在prefix后面生成新字符
    Defined in :numref:`sec_rnn_scratch`"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: d2l.reshape(d2l.tensor(
        [outputs[-1]], device=device), (1, 1))
    for y in prefix[1:]:  # 预热期
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # 预测num_preds步
        y, state = net(get_input(), state)
        # outputs.append(int(y.argmax(dim=1).reshape(1)))
        outputs.append(y)
    # return ''.join([vocab.idx_to_token[i] for i in outputs])
    return y

# read question and ground truth
def read_question():
    with open('/home/xwang/project/ACL10WP/data/predictionset.csv', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [line.strip().lower() for line in lines]

def read_answer():
    with open('/home/xwang/project/ACL10WP/data/groundtruth.csv', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [line.strip().lower() for line in lines]



if __name__ == "__main__":
    # read corpus
    lines = read_time_machine()
    print(f'# 文本总行数: {len(lines)}')
    # further tokenization
    tokens = tokenize(lines)
    # take a look at vocab
    vocab = Vocab(tokens)
    print(list(vocab.token_to_idx.items())[:10])

    # hyper-parameters and init
    batch_size, num_steps = 32, 35
    train_iter, vocab = load_data_time_machine(batch_size, num_steps)
    num_hiddens = 256
    rnn_layer = nn.RNN(len(vocab), num_hiddens)
    state = torch.zeros((1, batch_size, num_hiddens))
    X = torch.rand(size=(num_steps, batch_size, len(vocab)))
    Y, state_new = rnn_layer(X, state)

    # build net and use GPU
    device = d2l.try_gpu(0)
    net = RNNModel(rnn_layer, vocab_size=len(vocab))
    net = net.to(device)

    # training
    num_epochs, lr = 500, 1
    train_ch8(net, train_iter, vocab, lr, num_epochs, device)

    # test
    d2l.predict_ch8('我们 的', 10, net, vocab, device)

    # prepare prediction set
    m = torch.nn.Softmax(dim=1)
    questions = read_question()

    predix_candidate = []
    for sentence in questions:
        sentence = sentence.split("[mask]")
        predix_candidate.append(sentence[0])

    predix_as_context = []
    for sentence in predix_candidate:
        sentence = sentence.split(",")
        predix_as_context.append(sentence[0:-1])

    answers = read_answer()
    word_ground = []
    for tuples in answers:
        tuples = tuples.split(",")
        word_ground.append(tuples[0])

    result = []
    cur_for_answer = 0
    for predix_candidate in predix_as_context:
        prob_result = predict_probe(predix_candidate, 1, net, vocab, device)
        prob = m(prob_result)
        find_index = vocab[word_ground[cur_for_answer]]
        cur_for_answer += 1
        prob_number = prob[0][find_index]
        result.append(prob_number)

    final_result = []
    for number in result:
        number = number.detach().cpu().numpy()
        final_result.append(number)

    pd.DataFrame(final_result).to_csv("/home/xwang/project/ACL10WP/result/subtitle_RNN.csv", index=False)