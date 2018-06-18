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
            out_html += '<span style="color:' + ('red' if gold_ids[i][j] >= 2004 else 'green') + '; width:30px;">{:d} ({:d})</span>'.format(pred_ids[i][j],gold_ids[i][j]) + '</small></span>&nbsp;'
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

def output_eval(title,pred_tokens, pred_ids, pred_lens, gold_tokens, gold_lens, context, context_len, answer, answer_len):
    out_str ="""<html>
    <head><script type="text/javascript" language="javascript">
    function highlightCtxt(b,i)
    {
        document.getElementById('ctxt-'+b+'-'+i).className = "highlight";
    }
    function cancelHighlight(b,i)
    {
        document.getElementById('ctxt-'+b+'-'+i).className = ""
    }
    </script>
    <style type="text/css">.highlight{background:red;color:white;}</style></head><body>"""
    out_str+="<h1>Eval: "+title+' - '+str(datetime.datetime.now())+'</h1>'

    for b, pred in enumerate(pred_tokens):
        out_str+="<p>"
        for i,tok in enumerate(context[b][:context_len[b]-1].tolist()):
            out_str+='<span id="ctxt-'+str(b)+'-'+str(i)+'">'
            if i >=answer[b][0] and i < answer[b][0]+answer_len[b]:
                out_str+="<strong>"+tok.decode()+"</strong>"
            else:
                out_str+=tok.decode()
            out_str+='</span> '
        out_str+="</p>"
        pred_str=""
        for i,tok in enumerate(pred[:pred_lens[b]-1].tolist()):
            copy_id = None if pred_ids[b][i] < 2005 else pred_ids[b][i]-2004
            pred_str+='<span id="q-'+str(b)+'-'+str(i)+'" style="text-decoration:underline; text-decoration-color:'+ ("red" if pred_ids[b][i]>2004 else "green")+';"' +('onmouseover="highlightCtxt('+str(b)+','+str(copy_id)+');" onmouseout="cancelHighlight('+str(b)+','+str(copy_id)+');"' if copy_id is not None else '') +'>' \
            + tok.decode().replace('>','&gt;').replace('<','&lt;')+"</span> "
        gold_str = tokens_to_string(gold_tokens[b][:gold_lens[b]-1])

        out_str+="Pred: <pre>" +pred_str+"</pre><br/>Gold: <pre>"+gold_str.replace('>','&gt;').replace('<','&lt;')+"</pre><hr/>"
    out_str +="</body></html>"
    return out_str

def tokens_to_string(tokens):
    return " ".join([tok.decode() for tok in tokens])  #.replace('>','&gt;').replace('<','&lt;')
