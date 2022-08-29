import pandas as pd
import jieba

def jieba_tokenizer(sten):
    sten = jieba.cut(sten, cut_all=False)
    try:
        sten_list = list(sten)
        return sten_list
    except Exception:
        print("Attribute Error: add empty token.")
        return ['']

def concat(tokenlist):
  tokenstr = ' '
  return tokenstr.join(tokenlist)

corpus_root =  '/raid/xwang/corpus/corpus.csv' # corpus dir.
corpus_token_root = '/raid/xwang/corpus/token_file.csv' # dir to hold the token file

df = pd.read_csv(corpus_root, names=['content', 'none'], on_bad_lines='skip')
wiki_corpus_token = df['content'].apply(jieba_tokenizer)

wiki_corpus_token = wiki_corpus_token.reset_index(name='tokens')
wiki_corpus_token = wiki_corpus_token['tokens'].apply(concat)
wiki_corpus_token.to_csv(corpus_token_root, index=False, header=False)
