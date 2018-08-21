import sys,os,json
sys.path.insert(0, "/Users/tom/Dropbox/msc-ml/project/src/")
sys.path.insert(0, "/home/tomhosking/webapps/qgen/qgen/src/")

from flask import Flask, current_app, request, redirect

import sqlite3
import numpy as np
from nltk.tokenize import TreebankWordTokenizer, sent_tokenize

# homedir='/home/tomhosking/webapps/qgen_scores/htdocs/'
homedir='./'

# Filter a complete context down to the sentence containing the start of the answer span
def filter_context(ctxt, char_pos, window_size=0, max_tokens=-1):
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
            start=max(0, ix-window_size)
            end = min(len(sents)-1, ix+window_size)
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

app = Flask(__name__)

@app.route("/")
def index():
    return redirect("/static/annotate.htm")

@app.route("/api/save_score")
def set_score():
    init()
    qid = request.args['qid']
    q = request.args['q']
    model_id = request.args['model_id']
    fluency = request.args['fluency']
    relevance = request.args['relevance']
    scorer_name = request.args['scorer_name']
    ip = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)

    try:
        with sqlite3.connect(homedir+'scores/scores.sqlite') as db:
            db.execute('''INSERT INTO scores(qid, model_id, q, fluency, relevance, ip, scorer_name)
                      VALUES(?,?,?,?,?,?,?)''', (qid, model_id, q, fluency, relevance, ip, scorer_name))
    except sqlite3.IntegrityError:
        print('Record already exists')

    return json.dumps({'status':'success'})

@app.route("/api/get_q")
def get_q():
    init()
    scorer_name = request.args['scorer_name']
    ip = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)

    with sqlite3.connect(homedir+'scores/scores.sqlite') as db:
        res = db.execute('''SELECT COUNT(q) as count FROM scores WHERE scorer_name = ?''', (scorer_name,))
        ip_count = res.fetchone()[0]
    model_choice = np.random.choice(len(app.model_ids))
    # model_choice = ip_count%len(app.model_ids)
    qdict=app.questions[model_choice][ip_count]
    qdict['c'], qdict['a_pos'] = filter_context(qdict['c'], qdict['a_pos'], window_size=1)
    qdict['c'] = qdict['c'][:qdict['a_pos']] + '<strong>' + qdict['c'][qdict['a_pos']:]
    qdict['c'] = qdict['c'][:qdict['a_pos']+8+len(qdict['a_text'])] + '</strong>' + qdict['c'][qdict['a_pos']+8+len(qdict['a_text']):]
    return json.dumps({'ip_count': ip_count, **qdict})

def init():
    # print('Spinning up AQ annotator app')
    app.model_ids = ['MALUUBA','MALUUBA-CROP-LATENT','MALUUBA-CROP-SMART-SET','MALUUBA_RL_BOTH-LATENT-SCHEDULE', 'MALUUBA_RL_DISC-LATENT-JOINT']
    app.questions=[]
    for i,model_id in enumerate(app.model_ids):
        app.questions.append([])
        with open(homedir+'results'+'/out_eval_'+model_id+'_human.json') as f:
            results = json.load(f)['results']
            app.questions[i].extend([{'qid': ix, 'model_id':model_id ,**el} for ix,el in enumerate(results)])
    try:
        # Creates or opens a file called mydb with a SQLite3 DB
        app.db = sqlite3.connect(homedir+'scores/scores.sqlite')
        # Get a cursor object
        app.cursor = app.db.cursor()
        # Check if table users does not exist and create it
        app.cursor.execute('''CREATE TABLE IF NOT EXISTS
                          scores(qid INTEGER, model_id TEXT, q TEXT,
                           fluency INTEGER, relevance INTEGER, ip TEXT, scorer_name TEXT,
                            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        # Commit the change
        app.db.commit()
    # Catch the exception
    except Exception as e:
        # Roll back any change if something goes wrong
        app.db.rollback()
        raise e
    finally:
        # Close the db connection
        app.db.close()

if __name__ == '__main__':
    init()
    with app.app_context():
        app.run(port=14045)
