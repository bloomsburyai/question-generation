import numpy as np
import string
import tensorflow as tf

from nltk.tokenize import TreebankWordTokenizer, sent_tokenize
use_nltk = True

from helpers.loader import OOV, PAD, EOS, SOS

# def get_2d_spans(text, tokenss):
#     spanss = []
#     cur_idx = 0
#     for tokens in tokenss:
#         spans = []
#         for token in tokens:
#             if text.find(token, cur_idx) < 0:
#                 print("Tokens: {}".format(tokens))
#                 print("Token: {}\n Cur_idx: {}\n {}".format(token, cur_idx, repr(text)))
#                 raise Exception()
#             cur_idx = text.find(token, cur_idx)
#             spans.append((cur_idx, cur_idx + len(token)))
#             cur_idx += len(token)
#         spanss.append(spans)
#     return spanss
#
#
# def get_word_span(context, wordss, start, stop):
#     spanss = get_2d_spans(context, wordss)
#     idxs = []
#     for sent_idx, spans in enumerate(spanss):
#         for word_idx, span in enumerate(spans):
#             if not (stop <= span[0] or start >= span[1]):
#                 idxs.append((sent_idx, word_idx))
#
#     assert len(idxs) > 0, "{} {} {} {}".format(context, spanss, start, stop)
#     return idxs[0], (idxs[-1][0], idxs[-1][1] + 1)


def lookup_vocab(words, vocab, context=None, do_tokenise=True, append_eos=False, context_as_set=False, copy_priority=False):
    ids = []


    decoded_context = [w.decode() for w in tokenise(context)] if context is not None else []
    words = [w.decode() for w in tokenise(words)] if do_tokenise else [w.decode() for w in words]
    if context_as_set:
        context_set = sorted(set(decoded_context))

    for w in words:
        if copy_priority:
            if context is not None and not context_as_set and w in decoded_context:
                ids.append(len(vocab) + decoded_context.index(w))
            elif context is not None and context_as_set and w in context_set:
                ids.append(len(vocab) + context_set.index(w))
                # print(len(context_set), len(vocab) + context_set.index(w))
            elif w in vocab.keys():
                ids.append(vocab[w])
            else:
                ids.append(vocab[OOV])
        else:
            if w in vocab.keys():
                ids.append(vocab[w])
            elif context is not None and not context_as_set and w in decoded_context:
                ids.append(len(vocab) + decoded_context.index(w))
            elif context is not None and context_as_set and w in context_set:
                ids.append(len(vocab) + context_set.index(w))
                # print(len(context_set), len(vocab) + context_set.index(w))
            else:
                ids.append(vocab[OOV])
    if append_eos:
        ids.append(vocab[EOS])
    embedded = np.asarray(ids, dtype=np.int32)

    return embedded

# def find_start(haystack, key):
#     haystack = [w.decode() for w in haystack]
#     key = [w.decode() for w in key]
#     expanded = [haystack[i:i+len(key)] for i in range(0,len(haystack)-len(key)+1)]
#
#     if key in expanded:
#         return expanded.index(key)
#     else:
#         # TODO: handle this error - it shouldn't arise if the dataset is well formed and correctly tokenised
#         print(haystack)
#         print(key)
#         return expanded.index(key)

def tokenise(text, asbytes=True, append_eos=False):

    text = text.decode() if asbytes else text
    if use_nltk:
        sents = [s for s in sent_tokenize(text)]

        tokens = [tok.lower() for sent in sents for tok in TreebankWordTokenizer().tokenize(sent)]
    else:
        for char in string.punctuation+'()-–':
            text = text.replace(char, ' '+char+' ')
        tokens = text.lower().split(' ')
    tokens = [w.encode() if asbytes else w for w in tokens if w.strip() != '']
    if append_eos:
        tokens.append(EOS.encode() if asbytes else EOS)
    # tokens = np.asarray(tokens)
    # return np.asarray(tokens)
    return tokens

def char_pos_to_word(text, tokens, char_pos):
    ix=0
    text=text.decode()
    if use_nltk:
        sents = [s for s in sent_tokenize(text)]
        spans = [[s for s in TreebankWordTokenizer().span_tokenize(sent)] for sent in sents]
        # lens = [len(sent)+1  for sent in sents]
        offsets = [text.find(sent) for sent in sents] # can we do this faster?
        spans = [(span[0]+offsets[i], span[1]+offsets[i]) for i,sent in enumerate(spans) for span in sent]
        # print(char_pos)
        for ix,s in enumerate(spans):
            # print(s, tokens[ix])
            if s[1] > char_pos:
                return ix
        print('couldnt find the char pos via nltk')
        print(text, char_pos, len(text))
    else:
        tokens = [t.decode() for t in tokens]
        if char_pos>len(text):
            print('Char pos doesnt fall within size of text!')

        for t,token in enumerate(tokens):
            for char in token:
                ix = text.find(char, ix)
                ix += 1
                if ix >= char_pos:
                    return t
        print('couldnt find the char pos')
        print(text, tokens, char_pos, len(text))


def process_squad_context(vocab, context_as_set=False):
    def _process_squad_context(context):
        # print(context)
        # print(tokenise(context))
        context_ids = lookup_vocab(context, vocab, context=context, append_eos=True, context_as_set=context_as_set, copy_priority=True)
        context_set = set([w.decode() for w in tokenise(context)])

        context_len = np.asarray(len(context_ids), dtype=np.int32)
        context_vocab_size = np.asarray(len(context_set) if context_as_set else len(context_ids), dtype=np.int32)

        res = [tokenise(context,append_eos=True), context_ids, context_len, context_vocab_size]
        return res

    return _process_squad_context

def process_squad_question(vocab, context_as_set=False):
    def _process_squad_question(question, context):
        question_ids = lookup_vocab(question, vocab, context=context, append_eos=True, context_as_set=context_as_set)
        question_len = np.asarray(len(question_ids), dtype=np.int32)
        return [tokenise(question,append_eos=True), question_ids, question_len]
    return _process_squad_question

def process_squad_answer(vocab, context_as_set=False):
    def _process_squad_answer(answer, answer_pos, context):
        answer_ids = lookup_vocab(answer, vocab, context=context, append_eos=False, context_as_set=context_as_set)
        answer_len = np.asarray(len(answer_ids), dtype=np.int32)
        max_len = np.amax(answer_len)

        answer_token_pos=np.asarray(char_pos_to_word(context, tokenise(context), answer_pos), dtype=np.int32)

        answer_locs = np.arange(answer_token_pos, answer_token_pos+max_len, dtype=np.int32)

        return [tokenise(answer,append_eos=False), answer_ids, answer_len, answer_locs]
    return _process_squad_answer
