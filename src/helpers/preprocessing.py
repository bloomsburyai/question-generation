import numpy as np
import string
# import tensorflow as tf

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


def lookup_vocab(words, vocab, context=None, ans_tok_pos=None, do_tokenise=True, append_eos=False, context_as_set=False, copy_priority=False, asbytes=True, smart_copy=True, find_all=False ):
    ids = []

    decoded_context = [w.decode() if asbytes else w for w in tokenise(context)] if context is not None else []
    words = [w.decode() if asbytes else w for w in tokenise(words)] if do_tokenise else [w.decode() if asbytes else w for w in words]
    if context_as_set:
        context_set = sorted(set(decoded_context))

    for w in words:
        # Use a few heuristics to decide where to copy from
        if find_all:
            this_ids=[]
            if context is not None and not context_as_set and w in decoded_context:
                indices = [i+len(vocab) for i, x in enumerate(decoded_context) if x == w]
                this_ids.extend(indices)
            if context is not None and context_as_set and w in context_set:
                this_ids.append(len(vocab) + context_set.index(w))
            if w in vocab.keys():
                this_ids.append(vocab[w])
            if len(this_ids) ==0 :
                this_ids.append(vocab[OOV])
            ids.append(this_ids)
        elif copy_priority and smart_copy:
            if context is not None and not context_as_set and w in decoded_context:
                if decoded_context.count(w) > 1 and ans_tok_pos is not None:
                    # Multiple options, either pick the one that flows from previous, or pick the nearest to answer
                    if len(ids) > 0 and ids[-1]>=len(vocab) and len(decoded_context)>=ids[-1]-len(vocab)+2 and decoded_context[ids[-1]-len(vocab)+1] == w:
                        copy_ix = ids[-1]-len(vocab)+1
                    else:
                        indices = [i for i, x in enumerate(decoded_context) if x == w]
                        distances = [abs(ix-ans_tok_pos) for ix in indices]
                        copy_ix=indices[np.argmin(distances)]
                else:
                    copy_ix = decoded_context.index(w)
                ids.append(len(vocab) + copy_ix)
            elif context is not None and context_as_set and w in context_set:
                ids.append(len(vocab) + context_set.index(w))
            elif w in vocab.keys():
                ids.append(vocab[w])
            else:
                ids.append(vocab[OOV])
        # Copy using first occurence
        elif copy_priority:
            if context is not None and not context_as_set and w in decoded_context:
                ids.append(len(vocab) + decoded_context.index(w))
            elif context is not None and context_as_set and w in context_set:
                ids.append(len(vocab) + context_set.index(w))
                # print(len(context_set), len(vocab) + context_set.index(w))
            elif w in vocab.keys():
                ids.append(vocab[w])
            else:
                ids.append(vocab[OOV])
        # Shortlist priority
        else:
            if w in vocab.keys():
                ids.append(vocab[w])
            elif context is not None and not context_as_set and w in decoded_context:
                if smart_copy and decoded_context.count(w) > 1 and ans_tok_pos is not None:
                    # Multiple options, either pick the one that flows from previous, or pick the nearest to answer
                    if len(ids) > 0 and ids[-1]>=len(vocab) and len(decoded_context)>=ids[-1]-len(vocab)+2 and decoded_context[ids[-1]-len(vocab)+1] == w:
                        copy_ix = ids[-1]-len(vocab)+1
                    else:
                        indices = [i for i, x in enumerate(decoded_context) if x == w]
                        distances = [abs(ix-ans_tok_pos) for ix in indices]
                        copy_ix=indices[np.argmin(distances)]
                else:
                    copy_ix = decoded_context.index(w)
                ids.append(len(vocab) + copy_ix)
            elif context is not None and context_as_set and w in context_set:
                ids.append(len(vocab) + context_set.index(w))
                # print(len(context_set), len(vocab) + context_set.index(w))
            else:
                ids.append(vocab[OOV])
    if append_eos:
        ids.append(vocab[EOS] if not find_all else [vocab[EOS]])

    if not find_all:
        return np.asarray(ids, dtype=np.int32)
    else:
        return ids


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

def char_pos_to_word(text, tokens, char_pos, asbytes=True):
    ix=0
    text=text.decode() if asbytes else text
    if use_nltk:
        sents = [s for s in sent_tokenize(text)]
        spans = [[s for s in TreebankWordTokenizer().span_tokenize(sent)] for sent in sents]
        # lens = [len(sent)+1  for sent in sents]
        offsets = []
        for i,sent in enumerate(sents):
            offsets.append(text.find(sent, offsets[i-1]+len(sents[i-1]) if i>0 else 0)) # can we do this faster?
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

# Filter a complete context down to the sentence containing the start of the answer span
def filter_context(ctxt, char_pos, window_size_before=0, window_size_after=0, max_tokens=-1):
    sents = [s for s in sent_tokenize(ctxt)]
    spans = [[s for s in TreebankWordTokenizer().span_tokenize(sent)] for sent in sents]
    # lens = [len(sent)+1  for sent in sents]
    offsets = []
    for i,sent in enumerate(sents):
        # print(ctxt.find(sent, offsets[i-1]+len(sents[i-1]) if i>0 else 0))
        # print(len(sents[i-1]) if i>0 else 0)
        # print(offsets[i-1] if i>0 else 0)
        # print(offsets[i-1]+len(sents[i-1]) if i>0 else 0)
        offsets.append(ctxt.find(sent, offsets[i-1]+len(sents[i-1]) if i>0 else 0)) # can we do this faster?
    spans = [[(span[0]+offsets[i], span[1]+offsets[i]) for span in sent] for i,sent in enumerate(spans) ]
    for ix,sent in enumerate(spans):
        # print(sent[0][0], sent[-1][1], char_pos)
        if char_pos >= sent[0][0] and char_pos < sent[-1][1]:
            start=max(0, ix-window_size_before)
            end = min(len(sents)-1, ix+window_size_after)
            # print(start, end, start, offsets[start])
            # new_ix=char_pos-offsets[start]
            # print(new_ix)
            # print(" ".join(sents[start:end+1])[new_ix:new_ix+10])
            flat_spans=[span for sen in spans for span in sen]
            if max_tokens > -1 and len([span for sen in spans[start:end+1] for span in sen]) > max_tokens:
                for i,span in enumerate(flat_spans):
                    if char_pos < span[1]:
                        tok_ix =i
                        # print(span, char_pos)
                        break
                start_ix = max(spans[start][0][0], flat_spans[max(tok_ix-max_tokens,0)][0])
                end_ix = min(spans[end][-1][1], flat_spans[min(tok_ix+max_tokens, len(flat_spans)-1)][1])

                # if len(flat_spans[start_tok:end_tok+1]) > 21:
                # print(start_tok, end_tok, tok_ix)
                # print(flat_spans[tok_ix])
                # print(flat_spans[start_tok:end_tok])
                # print(ctxt[flat_spans[start_tok][0]:flat_spans[end_tok][1]])
                return ctxt[start_ix:end_ix], char_pos-start_ix
            else:
                return " ".join(sents[start:end+1]), char_pos - offsets[start]
    print('couldnt find the char pos')
    print(ctxt, char_pos, len(ctxt))

def filter_squad(data, window_size_before=0, window_size_after=0, max_tokens=-1):
    filtered=[]
    for row in data:
        filt_ctxt,new_ix = filter_context(row[0],row[3], window_size_before, window_size_after, max_tokens)
        filtered.append( (filt_ctxt, row[1],row[2],new_ix) )
    return filtered

def process_squad_context(vocab, context_as_set=False):
    def _process_squad_context(context):
        # print(context)
        # print(tokenise(context))
        context_ids = lookup_vocab(context, vocab, context=context, append_eos=True, context_as_set=context_as_set, copy_priority=False)
        context_copy_ids = lookup_vocab(context, vocab, context=context, append_eos=True, context_as_set=True, copy_priority=True)
        context_set = set([w.decode() for w in tokenise(context)])

        context_len = np.asarray(len(context_ids), dtype=np.int32)
        context_vocab_size = np.asarray(len(context_set) if context_as_set else len(context_ids), dtype=np.int32)

        res = [tokenise(context,append_eos=True), context_ids, context_copy_ids, context_len, context_vocab_size]
        return res

    return _process_squad_context

def process_squad_question(vocab, max_copy_size, context_as_set=False, copy_priority=False, smart_copy=True, latent_switch=False):
    def _process_squad_question(question, context, ans_loc):
        ans_tok_pos=char_pos_to_word(context, tokenise(context), ans_loc)
        question_ids = lookup_vocab(question, vocab, context=context, ans_tok_pos=ans_tok_pos, append_eos=True, context_as_set=context_as_set, copy_priority=copy_priority, smart_copy=smart_copy)
        question_len = np.asarray(len(question_ids), dtype=np.int32)
        if latent_switch:
            all_ids = lookup_vocab(question, vocab, context=context, ans_tok_pos=ans_tok_pos, append_eos=True, context_as_set=context_as_set, copy_priority=copy_priority, smart_copy=smart_copy, find_all=True)
            # print(all_ids)
            question_oh = np.asarray([np.sum(np.eye(len(vocab)+max_copy_size, dtype=np.float32)[ids], axis=0) for ids in all_ids], dtype=np.float32)
            # print(np.shape(np.eye(len(vocab)+max_copy_size, dtype=np.float32)[all_ids[0]]))
            # print(all_ids)
            # print(np.shape(question_oh))
            # exit()
        else:
            question_oh = np.eye(len(vocab)+max_copy_size, dtype=np.float32)[question_ids]
        return [tokenise(question,append_eos=True), question_ids, question_oh, question_len]
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
