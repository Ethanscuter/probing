import pandas as pd
import jieba
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import KneserNeyInterpolated
import nltk
from nltk.util import ngrams
from nltk.lm import Vocabulary
'''
Func. needed in DataFrames' Apply func.
'''


def jieba_tokenizer(sten):
    sten = jieba.cut(sten, cut_all=False)
    try:
        sten_list = list(sten)
        return sten_list
    except Exception:
        print("Attribute Error: add empty token.")
        return ['']


def candidate_bigram(mask_sent):
    mask_sent_list = list(mask_sent)
    mask_index = mask_sent_list.index('[MASK]')

    n_gram_candidates = [mask_sent_list[mask_index-1]]

    return n_gram_candidates


def candidate_trigram(mask_sent):
    mask_sent_list = list(mask_sent)
    mask_index = mask_sent_list.index('[MASK]')

    n_gram_candidates = [mask_sent_list[mask_index-2], mask_sent_list[mask_index-1]]

    return n_gram_candidates


def compute_score(model, candidate, ans):
    score = model.score(ans, candidate)
    return score


'''
Func. used in main func.
'''


def data_test_pre(predict_sent, answer, n):
    # test sentences and answer
    df_questions = pd.read_csv(predict_sent, header=None)
    df_answer = pd.read_csv(answer, header=None, names=['ans', 'value'])

    if n == 2:
        df_candidate = df_questions.apply(candidate_bigram, axis=1)
    elif n == 3:
        df_candidate = df_questions.apply(candidate_trigram, axis=1)
    df_ans = df_answer['ans']

    df_test = pd.DataFrame(list(zip(df_candidate, df_ans)), columns=['candidate', 'ans'])

    return df_test


def n_gram_basic(train_data, padded_sents, n):
    model = MLE(n)  # Lets train a 3-grams model, previously we set n=3
    model.fit(train_data, padded_sents)
    return model


def n_gram_smoothing(train_data, padded_sents, smooth_method, n):
    model_lap = KneserNeyInterpolated(n)
    model_lap.fit(train_data, padded_sents)
    return model_lap


def compute_predict_score(model, df_test):
    score = []
    for i in range(len(df_test)):
        current_score = compute_score(model, df_test['candidate'].iloc[i], df_test['ans'].iloc[i])
        score.append(current_score)

    return score


def until_write_to_file(score_list, result_location):
    df = pd.DataFrame(score_list)
    df.to_csv(result_location, index=False, header=False)


def build_vocab(corpus):
    words = []
    for word in corpus:
        words.extend(word)
    vocab = Vocabulary(words, unk_cutoff=2)
    return vocab


def filter_corpus(corpus, vocab):
    for sent_index, sent in enumerate(corpus):
        for word_index, word in enumerate(sent):
            if word not in vocab:
                corpus[sent_index][word_index] = 'UNK'
    return corpus


def check_test_token(elm1, elm2, elm3, vocab):
    if elm1 not in vocab:
        elm1 = 'UNK'
    if elm2 not in vocab:
        elm2 = 'UNK'
    if elm3 not in vocab:
        elm3 = 'UNK'
    return (elm1, elm2, elm3)


if __name__ == "__main__":
    # Define files' location.
    corpus,  predict_sent, answer = '../data/wiki_all_sub.csv', '../data/questionwiki.csv', '../data/answerwiki.csv'
    df_test = data_test_pre(corpus, predict_sent, answer, n=3)

    # n_gram prep.
    df = pd.read_csv(corpus, names=['content', 'none'])
    wiki_corpus = list(df['content'].apply(jieba_tokenizer))
    wiki_vocab = build_vocab(wiki_corpus)
    wiki_corpus = filter_corpus(wiki_corpus, wiki_vocab)
    gut_ngrams = (ngram for sent in wiki_corpus for ngram in
                  ngrams(sent, 3, pad_left=True, pad_right=True, right_pad_symbol='EOS', left_pad_symbol="BOS"))
    freq_dist = nltk.FreqDist(gut_ngrams)
    smooth_ngram = nltk.KneserNeyProbDist(freq_dist)

    # Compute score on test set
    score_list = []
    for i in range(len(df_test)):
        # tuple4kn = (df_test['candidate'].iloc[i][0], df_test['candidate'].iloc[i][1], df_test['ans'].iloc[i])
        tuple4kn = check_test_token(df_test['candidate'].iloc[i][0], df_test['candidate'].iloc[i][1], df_test['ans'].iloc[i], wiki_vocab)
        prob = smooth_ngram.prob(tuple4kn)
        if prob > 0.0:
            score_list.append(prob)
        else:
            score_list.append(smooth_ngram.prob(('UNK', 'UNK', 'UNK')))

    # Wrtie to file.
    result_location = '../' + 'result-o/' + 'smooth_ngram_' + 'KN' + '.csv'
    until_write_to_file(score_list, result_location)
