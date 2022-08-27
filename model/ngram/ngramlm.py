import pandas as pd
import jieba
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline

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


def data_train_test_pre(corpus, predict_sent, answer, n):
    # train and vocab
    df = pd.read_csv(corpus, names=['content', 'none'])
    wiki_corpus = list(df['content'].apply(jieba_tokenizer))
    train_data, padded_sents = padded_everygram_pipeline(n, wiki_corpus)

    # test sentences and answer
    df_questions = pd.read_csv(predict_sent, header=None)
    df_answer = pd.read_csv(answer, header=None, names=['ans', 'value'])

    if n == 2:
        df_candidate = df_questions.apply(candidate_bigram, axis=1)
    elif n == 3:
        df_candidate = df_questions.apply(candidate_trigram, axis=1)
    df_ans = df_answer['ans']

    df_test = pd.DataFrame(list(zip(df_candidate, df_ans)), columns=['candidate', 'ans'])

    return train_data, padded_sents, df_test


def n_gram_basic(train_data, padded_sents, n):
    model = MLE(n)  # Lets train a 3-grams model, previously we set n=3
    model.fit(train_data, padded_sents)
    return model


def compute_predict_score(model, df_test):
    score = []
    for i in range(len(df_test)):
        current_score = compute_score(model, df_test['candidate'].iloc[i], df_test['ans'].iloc[i])
        score.append(current_score)

    return score


def until_write_to_file(score_list, result_location):
    df = pd.DataFrame(score_list)
    df.to_csv(result_location, index=False, header=False)


if __name__ == "__main__":
    # Define files' location. question is a mask file of behavioral data, answerwiki is the <word, prob.>
    corpus,  predict_sent, answer = '../data/wiki_all_sub.csv', '../data/questionwiki.csv', '../data/answerwiki.csv'

    # Train basic n_gram models, n is from 2 to 3
    min_N, max_N = 3, 3
    for n in range(min_N, max_N+1):
        # Train, test sets prepare.
        train_data, padded_sents, df_test = data_train_test_pre(corpus, predict_sent, answer, n)
        # Train a basic n_gram model whose N is the current n.
        print('Current N-gram model is basic model, N is : ' + str(n))
        basic_ngram = MLE(n)  # Lets train a 3-grams model, previously we set n=3
        basic_ngram.fit(train_data, padded_sents)

        # Compute score on test set
        score_list = compute_predict_score(basic_ngram, df_test)
        print(score_list)

        # Wrtie to file.
        result_location = '../' + 'result-o' + '_basic_ngram' + str(n) + '.csv'
        until_write_to_file(score_list, result_location)
