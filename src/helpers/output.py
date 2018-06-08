import datetime

def output_pretty(tokens, pred_ids, gold_ids, copy,e,i):
    out_html = '''<style type="text/css">small {
    position: absolute;
    top: 18px;
    float: left;
    left: -0;
    font-size: 10px;
    }
    span.word{
    margin: 0 0 35px 0;
    display: inline-block;
    position: relative;
    min-width:40px;
    }
    small span { padding:0px;}
    </style>'''
    out_html+='<h1>'+str(e)+', '+str(i)+' ('+ str(datetime.datetime.now()) +')</h1>'
    for i,row in enumerate(tokens):
        out_html += '<p>'
        for j, tok in enumerate(row):
            out_html += '<span class="word">' + tok.decode().replace('>','&gt;').replace('<','&lt;')  +'<small>'
            out_html += '<span style="background:red; display:inline-block; width:' +str(copy[i][j]*30)+'px; height:8px;">&nbsp;</span><span style="background:green; display:inline-block; width:' +str((1-copy[i][j])*30) +'px; height:8px;">&nbsp;</span><br/>'
            out_html += '<span style="color:' + ('red' if gold_ids[i][j] > 2004 else 'green') + '; width:30px;">{:d} ({:d})</span>'.format(pred_ids[i][j],gold_ids[i][j]) + '</small></span>&nbsp;'
        out_html += '</p>'
    return out_html


def output_basic(tokens, ids, copy, shortlist, epoch, step_num):
    out_html='<h1>'+str(epoch)+', '+str(step_num)+' ('+ str(datetime.datetime.now()) +')</h1>'
    for i,row in enumerate(tokens):
        out_html += '<p>'
        for j, tok in enumerate(row):
            out_html += tok.decode().replace('>','&gt;').replace('<','&lt;') + '('+ str(ids[i][j])  +') '
            out_html += '<span style="background:red; display:inline-block; width:' +str(copy[i][j]*30)+'px;">&nbsp;</span><span style="background:green; display:inline-block; width:' +str(shortlist[i][j]*30) +'px;">&nbsp;</span>'
        out_html += '</p>'
    return out_html


def tokens_to_string(tokens):
    return " ".join([tok.decode() for tok in tokens])  #.replace('>','&gt;').replace('<','&lt;')
