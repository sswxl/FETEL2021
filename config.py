from platform import platform
from os.path import join
import socket

if platform().startswith('Windows'):
    PLATFORM = 'Windows'
    DATA_DIR = 'E:/Pycoding/biye/Biye2021/data/fetel-data'
else:
    PLATFORM = 'Linux'
    DATA_DIR = '/data/hldai/fetel-data'

TOKEN_UNK = '<UNK>'#0
TOKEN_ZERO_PAD = '<ZPAD>'#1
TOKEN_EMPTY_PAD = '<EPAD>'#2
TOKEN_MENTION = '<MEN>'#3

RANDOM_SEED = 771
NP_RANDOM_SEED = 7711
PY_RANDOM_SEED = 9973

MACHINE_NAME = socket.gethostname()
RES_DIR = join(DATA_DIR, 'res')
EL_DATA_DIR = join(DATA_DIR, 'el')
MODEL_DIR = join(DATA_DIR, 'models')
LOG_DIR = join(DATA_DIR, 'log')

EL_CANDIDATES_DATA_FILE = join(RES_DIR, 'enwiki-20151002-candidate-gen.pkl')
WIKI_FETEL_WORDVEC_FILE = join(RES_DIR, 'enwiki-20151002-nef-wv-glv840B300d.pkl')
WIKI_ANCHOR_SENTS_FILE = join(RES_DIR, 'enwiki-20151002-anchor-sents.txt')#error

FIGER_FILES = {
    'typed-wiki-mentions': join(DATA_DIR, 'Wiki/enwiki-20151002-anchor-mentions-typed.txt'),#没有这个文件
    'anchor-train-data-prefix': join(DATA_DIR, 'Wiki/enwiki20151002anchor-fetwiki-0_1'),
    'type-vocab': join(DATA_DIR, 'Wiki/figer-type-vocab.txt'),
    #{"wid": 41529999, "types": ["/person"]}
    'wid-type-file': join(DATA_DIR, 'Wiki/wid-types-figer.txt'),
    #{"mention_id": 20, "span": [0, 1], "str": "Hopkins", "sent_id": 12, "labels": ["/person"], "file_id": "0"}
    'fetel-test-mentions': join(DATA_DIR, 'Wiki/figer-fetel-test-mentions.json'),
    #{"sent_id": 45, "text": "Investigators suspect Cho is in the Los Angeles area .", "afet-senid": 45, "file_id": "1"}
    'fetel-test-sents': join(DATA_DIR, 'Wiki/figer-fetel-test-sents.json'),
    #{"mention_id": 7, "labels": ["/organization"], "logits": [-3.4627645015716553, -1.8523982763290405,......
    'noel-typing-results': join(DATA_DIR, 'Wiki/noel-fet-results-aaa-figer.txt')
}

BBN_FILES = {
    'anchor-train-data-prefix': join(DATA_DIR, 'BBN/enwiki20151002anchor-fetbbn-0_1'),
    'type-vocab': join(DATA_DIR, 'BBN/bbn-type-vocab.txt'),
    'wid-type-file': join(DATA_DIR, 'BBN/wid-types-bbn.txt'),
    'fetel-test-mentions': join(DATA_DIR, 'BBN/bbn-fetel-test-mentions.json'),
    'fetel-test-sents': join(DATA_DIR, 'BBN/bbn-fetel-test-sents.json'),
    'noel-typing-results': join(DATA_DIR, 'BBN/noel-fet-results-aaa-bbn.txt')
}
