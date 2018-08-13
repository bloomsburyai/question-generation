import sys,os,json
sys.path.insert(0, "/Users/tom/Dropbox/msc-ml/project/src/")
sys.path.insert(0, "/home/tomhosking/webapps/qgen/qgen/src/")

from flask import Flask, current_app, request, redirect

import sqlite3
import numpy as np

app = Flask(__name__)

@app.route("/")
def index():
    return redirect("/static/annotate.htm")

@app.route("/api/save_score")
def set_score():
    qid = request.args['qid']
    q = request.args['q']
    model_id = request.args['model_id']
    score = request.args['score']
    ip = request.remote_addr

    try:
        with sqlite3.connect('./scores/scores.sqlite') as db:
            db.execute('''INSERT INTO scores(qid, model_id, q, score, ip)
                      VALUES(?,?,?,?,?)''', (qid, model_id, q, score, ip))
    except sqlite3.IntegrityError:
        print('Record already exists')

    return json.dumps({'status':'success'})

@app.route("/api/get_q")
def get_q():
    ip = request.remote_addr
    with sqlite3.connect('./scores/scores.sqlite') as db:
        res = db.execute('''SELECT COUNT(score) as count FROM scores WHERE ip = ?''', (ip,))
        ip_count = res.fetchone()[0]
    qdict = np.random.choice(app.questions)
    return json.dumps({'ip_count': ip_count, **qdict})

def init():
    print('Spinning up AQ annotator app')
    model_ids = ['MALUUBA-CROP-LATENT','BASELINE','MALUUBA']
    app.questions=[]
    for model_id in model_ids:
        with open('./results'+'/out_eval_'+model_id+'.json') as f:
            results = json.load(f)['results']
            app.questions.extend([{'qid': ix, 'model_id':model_id ,**el} for ix,el in enumerate(results)])
    try:
        # Creates or opens a file called mydb with a SQLite3 DB
        app.db = sqlite3.connect('./scores/scores.sqlite')
        # Get a cursor object
        app.cursor = app.db.cursor()
        # Check if table users does not exist and create it
        app.cursor.execute('''CREATE TABLE IF NOT EXISTS
                          scores(qid INTEGER, model_id TEXT, q TEXT, score INTEGER, ip TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
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
