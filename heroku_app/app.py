import flask
import requests
import time
import dill
import pandas as pd
import numpy as np
from collections import OrderedDict
from datetime import date,datetime
from flask import request, session,Flask, Markup, url_for
from flask_table import Table, Col, LinkCol
from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.resources import INLINE
from bokeh.templates import JS_RESOURCES, CSS_RESOURCES
from bokeh.util.string import encode_utf8
from bokeh.models import HoverTool, ColumnDataSource
from sklearn.base import BaseEstimator,RegressorMixin,TransformerMixin
from bokeh.models import LinearAxis, Range1d


app = flask.Flask(__name__)
app.secret_key = '\x87\x1e2\x89\x0f|"\xf9\xbbh\xabj\xd1\xf8\xaf\xc0\x06\xcd\x9e\xdb\xe5\xc6\xaf\xac'

with open('./List2DictT.pkl','rb') as in_strm:
    List2DictT = dill.load(in_strm)
with open('./DictVectorT.pkl','rb') as in_strm:
    DictVectorT = dill.load(in_strm)
with open('./FinalTotModes.pkl','rb') as in_strm:
    FinalTotModes = dill.load(in_strm)
with open('risk.pkl','rb') as in_strm:
    crit_r_c = dill.load(in_strm)

with open("./api.nogit",'r') as in_strm:
    secrets = (in_strm.read())

with open('MedianScore.pkl','rb') as in_strm:
    MedianScore = dill.load(in_strm)
with open('ScoreByGrade.pkl','rb') as in_strm:
    ScoreByGrade = dill.load(in_strm)

colors = ['#FF0000','#00FF00','#0000FF']
RequiredDict = {'sub_grade':'Lending Club Grade',
                'term':'Term',
                'fico_range_low': 'Fico Score (Low)',
                'fico_range_high':'Fico Score (High)',
                'loan_amnt':'Total Loan Amount',
                'issue_d':'Loan issued date',
                'annual_inc': 'Annual Income',
                'dti':'Debt to Income ratio',
                'emp_length':'Employment length',
                'open_acc':'Number of credit lines(open) ',
                'total_acc':'Total credit lines',
                'earliest_cr_line':'Earliest credit line ',
                'pub_rec':'Public record',
                'policy_code':'Polic code',
                'home_ownership':'House Ownership'}
RequiredArgs = ['sub_grade', 'term', 'fico_range_low', 'fico_range_high', 'loan_amnt', 'issue_d',
                'annual_inc', 'dti', 'emp_length', 'open_acc', 'total_acc',
                'earliest_cr_line', 'pub_rec', 'policy_code', 'home_ownership']

fieldOI = ['id','Prediction','PayOff_live','subgrade','TopPct_live',
           'term','ficorangehigh','ficorangelow','memberid','url']
relevant =['annual_inc', 'dti', 'emp_length', 'fico_range_high', 'fico_range_low', 'home_ownership',
           'loan_amnt', 'open_acc', 'policy_code', 'pub_rec', 'sub_grade', 'term', 'total_acc',
           'earliest_cr_line', 'issue_d']        

class cLinkCol(LinkCol):
    def outurl(self, item):
        return item['link']
    def td_contents(self, item, attr_list):
        return '<a href="{url}">{text}</a>'.format(
            url=self.outurl(item),
            text=Markup.escape(self.text(item, attr_list)))

class cTable(Table):
    def classes_html_attr(self):
        if not self.classes:
            return ''
        else:
            return ' class="{}" id="myTable"'.format(' '.join(self.classes))

class SortableTable(cTable):
    classes = ['tablesorter']

    score, chance, grade =  Col('Score'), Col('Success(%)'), Col("LC Grade")
    top, term, high, low, id, memberid = Col("Top(%)"), Col("Term"), Col("FICO low"), Col("FICO high"), Col('Loan ID'), Col('Borrrower ID')
    link = cLinkCol(' Link ','link',url_kwargs=dict(id='id'),allow_sort=False)
    allow_sort = False


def inputQualitycheck(args):
    MissingArgs,error_Message = [], ''
    for RequiredArg in RequiredArgs:
        if RequiredArg not in args or not args[RequiredArg]:
            MissingArgs.append(RequiredArg)
    if not len(MissingArgs) == 0:
        for MissingArg in MissingArgs:
            error_Message += RequiredDict[MissingArg] + ', '
    if len(error_Message)>0:
        return (args, error_Message)    
    else:
        newargs ={}
        for key in args:
            if (key == 'issue_d') and \
            (args[key]).lower() in ['n.a.', 'n.a.','n/a','na','still alive','alive']:
                newargs[key] = date.today().strftime("%b-%Y")
            else:
                newargs[key] = (args[key]).strip()
        return (newargs, error_Message)

def getitem(obj, item, default):
    if item not in obj:
        return default
    else:
        return obj[item]

class List2Dict_Transformer(BaseEstimator,TransformerMixin):
    '''
    Expects a data-frame object. 
    '''
    def __init__(self, str_keys=[], live=0):
        self.str_keys = str_keys
        self.live = live
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        from datetime import datetime
        grademap = {'A':10,"B":20,"C":30,"D":40,"E":50,"F":60,"G":70,"H":80}
        X_dict = []
        X_keys = X.columns
        for i in xrange(len(X)):
            x_dict = {}
            for key in X_keys:
                if key in self.str_keys:
                    x_dict[key + '_' + X[key].iloc[i]] = 1
                elif key == u'emp_length':
                    if self.live == 0:
                        recode = X[key].iloc[i][0].lower()
                        if recode == 'n':
                            x_dict[key] = -1
                        elif recode == '<':
                            x_dict[key] = 0
                        else:
                            x_dict[key] = int(X[key].iloc[i][:2])
                    else:
                        if np.isnan(X[key].iloc[i]):
                            x_dict[key] = -1
                        else:
                            x_dict[key] = X[key].iloc[i] /12    
                elif key == 'sub_grade':
                    base = grademap[X[key].iloc[i][0]]
                    x_dict[key] = base + int(X[key].iloc[i][1])
                elif key == 'term' and self.live==0:
                    x_dict[key] = int(X[key].iloc[i][:3])
                elif key in ['issue_d', 'earliest_cr_line']:
                    if self.live==1 and key == 'earliest_cr_line':
                        x_dict[key] = datetime.strptime(X[key].iloc[i][:10],'%Y-%m-%d')
                    else:
                        x_dict[key] = datetime.strptime(X[key].iloc[i],'%b-%Y')
                else:
                    x_dict[key] = float(X[key].iloc[i])
            x_dict['CreditLength'] = (x_dict['issue_d'] - x_dict['earliest_cr_line']).days
            del x_dict['issue_d']
            del x_dict['earliest_cr_line']
            X_dict.append(x_dict)
        return X_dict

@app.route("/",methods=["GET","POST"])
def welcome():
    html = flask.render_template('welcome_w.html')
    return encode_utf8(html)


@app.route("/loan_input/",methods=["GET","POST"])
def loan_input():
    if request.method == 'GET':
        return flask.render_template('Input.html')
    elif request.method =='POST':
        args = flask.request.form
        (args, error_Message) = inputQualitycheck(args)
        if len(error_Message)>0:
            flask.flash(error_Message[:-2] + '!')
            return flask.redirect(flask.url_for('loan_input'))
        else:
            session['loan'] = args
            return flask.redirect('prediction')

@app.route("/loan_input_demo1/",methods=["GET","POST"])
def loan_input_demo1():
    if request.method == 'GET':
        return flask.render_template('Input_demo1.html')
    elif request.method =='POST':
        args = flask.request.form
        (args, error_Message) = inputQualitycheck(args)
        if len(error_Message)>0:
            flask.flash(error_Message[:-2] + '!')
            return flask.redirect(flask.url_for('loan_input'))
        else:
            session['loan'] = args
            return flask.redirect('prediction')

@app.route("/loan_input_demo2/",methods=["GET","POST"])
def loan_input_demo2():
    if request.method == 'GET':
        return flask.render_template('Input_demo2.html')
    elif request.method =='POST':
        args = flask.request.form
        (args, error_Message) = inputQualitycheck(args)
        if len(error_Message)>0:
            flask.flash(error_Message[:-2] + '!')
            return flask.redirect(flask.url_for('loan_input'))
        else:
            session['loan'] = args
            return flask.redirect('prediction')

@app.route("/loan_input_demo3/",methods=["GET","POST"])
def loan_input_demo3():
    if request.method == 'GET':
        return flask.render_template('Input_demo3.html')
    elif request.method =='POST':
        args = flask.request.form
        (args, error_Message) = inputQualitycheck(args)
        if len(error_Message)>0:
            flask.flash(error_Message[:-2] + '!')
            return flask.redirect(flask.url_for('loan_input'))
        else:
            session['loan'] = args
            return flask.redirect('prediction')

@app.route("/loan_input_demo4/",methods=["GET","POST"])
def loan_input_demo4():
    if request.method == 'GET':
        return flask.render_template('Input_demo4.html')
    elif request.method =='POST':
        args = flask.request.form
        (args, error_Message) = inputQualitycheck(args)
        if len(error_Message)>0:
            flask.flash(error_Message[:-2] + '!')
            return flask.redirect(flask.url_for('loan_input'))
        else:
            session['loan'] = args
            return flask.redirect('prediction')

@app.route("/loan_input_yourloan/",methods=["GET","POST"])
def loan_input_yourloan():
    if request.method == 'GET':
        return flask.render_template('Input_autofocus.html')
    elif request.method =='POST':
        args = flask.request.form
        (args, error_Message) = inputQualitycheck(args)
        if len(error_Message)>0:
            flask.flash(error_Message[:-2] + '!')
            return flask.redirect(flask.url_for('loan_input'))
        else:
            session['loan'] = args
            return flask.redirect('prediction')

@app.route('/prediction/',methods=["GET"])
def prediction():
    args = session['loan']
    Grade = args['sub_grade']

    x = pd.DataFrame(args,index=[0])
    x_dict = List2DictT.transform(x)

    x_vect = DictVectorT.transform(x_dict)
    x_y_prob = []
    for RF in FinalTotModes[:-1]:
        x_y_prob.append(RF.predict_proba(x_vect)[:,0])
    y_pred = FinalTotModes[-1].predict_proba(np.array(x_y_prob).T)
    YourLoan = int(y_pred[:,1]*100)
    Loc  = int(YourLoan/5)
    xgrades =[i for i in ScoreByGrade.index]

    Beats = ['%.2f' %(ScoreByGrade.PayoffProba.iloc[Loc]*100) + 
             ' (your loan is better than %.2f%% of loans)' % (sum(ScoreByGrade[Grade].iloc[:Loc]))]
    
    Median = MedianScore.loc[Grade].prob
    MedianLoc = MedianLoc = int(Median)/5+1

    TOOLS = "hover,pan,box_zoom,reset,save"
    p = figure(background_fill='white', 
               x_range=[2,102],
               x_axis_label='Score assigned by the model',
               y_range = [0, int(max(ScoreByGrade[Grade]))+3],
               title="Score of " + Grade + " loans",
               y_axis_label='Counts per 100 loans',
               tools = TOOLS,
               plot_width=1000, 
               plot_height=600)
    
    source1 = ColumnDataSource(
        data=dict(payoffProb = ScoreByGrade.PayoffProba.values*100, 
              distribution =  ScoreByGrade[Grade].values,
              scores = ['%d-%d' %(i-4,i) for i in ScoreByGrade.index]))
    source2 = ColumnDataSource(
        data=dict(payoffProb = Beats, distribution =[ScoreByGrade[Grade].iloc[Loc]],
                  scores = ['%d-%d' %(i-4,i) for i in [ScoreByGrade.index[Loc]]]))

    source3 = ColumnDataSource(
        data=dict(payoffProb = [ScoreByGrade.PayoffProba.iloc[MedianLoc]*100], 
                  distribution =[ScoreByGrade[Grade].iloc[MedianLoc]],
                  scores = ['%d' %(Median) ]))


    p.rect(xgrades, ScoreByGrade[Grade]/2,  0.6*5, ScoreByGrade[Grade],
        fill_color="#08c994",source = source1)
    p.rect([xgrades[Loc]], ScoreByGrade[Grade].iloc[Loc]/2,  0.6*5, ScoreByGrade[Grade].iloc[Loc],
        fill_color="#ff5a00",source = source2, legend='Score of your loan')
    # p.xaxis.major_label_orientation = np.pi/6
    p.rect([Median], (int(max(ScoreByGrade[Grade]))+1)/2.0,  0.15*5, (int(max(ScoreByGrade[Grade]))+1),
        fill_color="black",source = source3, legend='Average Score')

    # p.line([Median,Median],[0,int(max(ScoreByGrade[Grade]))+1],
    #         line_color="black", line_width=10, legend='Average Score',
    #         source = source3)


    hover = p.select(dict(type=HoverTool))
    hover.tooltips = OrderedDict([
        ("Score","@scores"),
        ("Within 100 loans", "@distribution in this group"),
        ('Payoff chance (%)', '@payoffProb'),
    ])

    js_resources = JS_RESOURCES.render(
        js_raw=INLINE.js_raw,
        js_files=INLINE.js_files
    )

    css_resources = CSS_RESOURCES.render(
        css_raw=INLINE.css_raw,
        css_files=INLINE.css_files
    )

    script, div = components(p, INLINE)

    if YourLoan>=60:
        ShouldOrNot = 'SHOULD'
        tmplt ="result_success.html"
    elif YourLoan>=50:
        ShouldOrNot = 'MAY'
        tmplt ="result_success.html"
    elif YourLoan>=40:
        ShouldOrNot = 'may NOT'
        tmplt ="result_failure.html"
    else:
        ShouldOrNot = 'should NOT'
        tmplt ="result_failure.html"

    pTOOLS = "crosshair,hover,pan,box_zoom,reset,save"
    p2 = figure(background_fill='white', 
               x_axis_label='Score assigned by the model',
               title="Score versus Success rate",
               y_axis_label='Success rate (%)',
               tools = pTOOLS,
               plot_width=1000, 
               plot_height=600)

    p2.line(ScoreByGrade.index,ScoreByGrade.PayoffProba*100,line_color="blue", line_width=20, alpha=0.7)
    r2 = p2.circle(Median,ScoreByGrade.PayoffProba.iloc[MedianLoc]*100-3,color="black",legend='Average Loan')
    r  = p2.circle(YourLoan,ScoreByGrade.PayoffProba.iloc[Loc]*100-3,color="#ff5a00",legend='Your Loan')
    def glyy(r,colors):
        glyph = r.glyph
        glyph.size = 40
        glyph.fill_alpha = 0.5
        glyph.line_color = colors
        glyph.line_dash = [6, 3]
        glyph.line_width = 2
    glyy(r,"#ff5a00")
    glyy(r2,"black")

    p2.grid.grid_line_alpha=0.7
    p2.legend.orientation = "top_left"
    hover2 = p2.select(dict(type=HoverTool))
    hover2.tooltips = OrderedDict([
        ('Score', "$x"),
        ('Payoff chance (%)', '$y')])

    js_resources2 = JS_RESOURCES.render(
        js_raw=INLINE.js_raw,
        js_files=INLINE.js_files)

    css_resources2 = CSS_RESOURCES.render(
        css_raw=INLINE.css_raw,
        css_files=INLINE.css_files)
    p2script, p2div = components(p2, INLINE)


    html = flask.render_template(
        tmplt,
        plot_script=script, plot_div=div, 
        plot_script2=p2script, plot_div2=p2div, 
        js_resources=js_resources,
        css_resources=css_resources,
        score=YourLoan,
        beats="%.2f" %(sum(ScoreByGrade[Grade].iloc[:Loc])),
        prob="%.2f" %(ScoreByGrade.PayoffProba.iloc[Loc]*100),
        invest=ShouldOrNot)

    return encode_utf8(html)


@app.route('/loanstats/')
def loanstats():
    html = flask.render_template('loanstats_tmplt.html')
    return encode_utf8(html)

@app.route('/live_loans/')
def live_loans():
    with open('liveloan_TS.pkl','rb') as in_strm:
        (pliveloans, TStmp) = dill.load(in_strm)

    nowTS = datetime.now()
    print TStmp, nowTS
    if (nowTS - TStmp).seconds>= 1*60*60:
        try:
            attempt = 5
            while attempt >0:
                print '\nTry to download loans from LC .... attempt: %d \n' %(6 - attempt)
                r = requests.get("https://api.lendingclub.com/api/investor/v1/loans/listing", 
                    headers={'Authorization':secrets},params={'showAll' : True})
                if r.status_code == 200:
                    print '\nLoans downloaded!\n'
                    break
                else:
                    time.sleep(1.1) 
                    attempt -= 1
            liveloans=pd.DataFrame.from_dict( r.json()['loans'] )
            liveloans.columns = [i.lower() for i in liveloans.columns]
            X_live_recorded = pd.DataFrame(columns=relevant)
            for key in relevant:
                if key == 'loan_amnt':
                    X_live_recorded[key] = liveloans['loanamount']
                elif key in ['policy_code', 'issue_d']:
                    pass
                else:
                    X_live_recorded[key] = liveloans[key.replace('_','')]
            
            X_live_recorded['policy_code'] = 1
            X_live_recorded['issue_d'] = datetime.today().strftime("%b-%Y")
            newList2DictT = List2Dict_Transformer(str_keys=['home_ownership'],live=1)
            X_live_trans = newList2DictT.transform(X_live_recorded)
            X_live_vect = DictVectorT.transform(X_live_trans)
            y_liveprob=[]
            for RF in FinalTotModes[:-1]:
                y_liveprob.append(RF.predict_proba(X_live_vect)[:,0])
            y_livefinalpred = FinalTotModes[-1].predict_proba(np.array(y_liveprob).T)
            LiveLoanPred = (y_livefinalpred[:,1]*100)
            liveloans['Prediction'] = LiveLoanPred
            
            Locs  = [int(liveloans['Prediction'].iloc[i]/5) for i in range(len(liveloans))]
            PayOff_live = [ScoreByGrade.PayoffProba.iloc[Loc]*100 for Loc in Locs]
            TopPct_live = np.array([100-sum(ScoreByGrade[liveloans.subgrade.iloc[i]].iloc[:Locs[i]+1]) for i in range(len(liveloans))])
            TopPct_live[TopPct_live<=0.01] =0.01
            liveloans['Locs'] = Locs
            liveloans['PayOff_live'] = PayOff_live
            liveloans['TopPct_live'] = TopPct_live
            baseurl = 'https://www.lendingclub.com/browse/loanDetail.action?loan_id='
            urls = [baseurl + str(liveloans.id.iloc[i]) for i in xrange(len(liveloans))]
            liveloans['url'] = urls
            TStmp = nowTS

            if len(liveloans)>10:
                with open('liveloan_TS.pkl','wb') as out_strm:
                    dill.dump((liveloans, nowTS),out_strm)
        except:
            print '\n Fail to download the newest loan, use last saved loan!\n'
            liveloans = pliveloans
            pass
    else:
        liveloans = pliveloans


    TableData = liveloans[fieldOI]
    TableData.columns=['id', 'score', 'chance', 'grade', 'top', 'term', 'high', 'low', 'memberid','link']
    TableData['score'] =  [int(TableData.score.iloc[i]) for i in xrange(len(TableData))]
    TableData['chance'] = ['%.2f' %(TableData.chance.iloc[i]) for i in xrange(len(TableData))]
    TableData['top'] = ['%.2f' %(TableData.top.iloc[i]) for i in xrange(len(TableData))]

    table = SortableTable(TableData.to_dict('records'))
    # print table.td_content
    TStrin=TStmp.strftime("%Y-%m-%d %H:%M:%S")
    html = flask.render_template('live_loan.html',
        table=table,
        TStrin=TStrin,
        nLoans= str(len(TableData)))

    return encode_utf8(html)

@app.route('/model/')
def model():
    pTOOLS = "crosshair,hover,pan,box_zoom,reset,save"
    p2 = figure(tools=pTOOLS,background_fill="#dbe0eb",
                x_range = (0,100),
                y_range =(80,100),
                x_axis_label='Decision Boundary (score)',
                y_axis_label='Precision',
                title='Recall and Precision',
                plot_height=600,
                plot_width=800)

    # Setting the second y axis range name and range
    p2.extra_y_ranges = {"foo": Range1d(start=-5, end=105)}

    # Adding the second axis to the plot.  
    p2.add_layout(LinearAxis(y_range_name="foo",axis_label="Recall", 
                             axis_label_text_color = "red",
                             major_label_text_color = "red"), 'right')

    source1 = ColumnDataSource(
        data=dict(precision = crit_r_c[1]*100, recall =  crit_r_c[2]*100.0/153546,
                 risk = 100 - crit_r_c[1]*100, miss = 100- crit_r_c[2]*100.0/153546))

    source2 = ColumnDataSource(
        data=dict(precision = crit_r_c[1]*100, recall =  crit_r_c[2]*100.0/153546,
                 risk = 100 - crit_r_c[1]*100, miss = 100- crit_r_c[2]*100.0/153546))


    p2.line(crit_r_c[0],crit_r_c[1]*100,line_color="blue", line_width=20, alpha=0.7,source=source1)
    p2.line(crit_r_c[0],crit_r_c[2]*100.0/153546,line_color="red", line_width=20, alpha=0.7,y_range_name="foo",source=source2)

    hover = p2.select(dict(type=HoverTool))
    hover.tooltips = OrderedDict([
        ('Decison Boundary (score)', '$x'),
        ("Precision            (%)","@precision"),
        ("Risk                 (%)", "@risk"),
        ("Recall               (%)", "@recall"),
        ("Missed Opportunity   (%)", "@miss"),

    ])

    js_resources2 = JS_RESOURCES.render(
        js_raw=INLINE.js_raw,
        js_files=INLINE.js_files)

    css_resources2 = CSS_RESOURCES.render(
        css_raw=INLINE.css_raw,
        css_files=INLINE.css_files)
    p2script, p2div = components(p2, INLINE)

    html = flask.render_template(
        'model.html',
        plot_script2=p2script, plot_div2=p2div, 
        js_resources=js_resources2,
        css_resources=css_resources2)
    return encode_utf8(html)

app.add_url_rule('/', 'index', welcome)
app.add_url_rule('/loan_input/', 'loan_input', loan_input)
app.add_url_rule('/loan_input_demo1/', 'loan_input_demo1', loan_input_demo1)
app.add_url_rule('/loan_input_demo2/', 'loan_input_demo2', loan_input_demo2)
app.add_url_rule('/loan_input_demo3/', 'loan_input_demo3', loan_input_demo3)
app.add_url_rule('/loan_input_demo4/', 'loan_input_demo4', loan_input_demo4)
app.add_url_rule('/loan_input_yourloan/', 'loan_input_yourloan', loan_input_yourloan)

app.add_url_rule('/prediction/', 'prediction', prediction)
app.add_url_rule('/loanstats/', 'loanstats', loanstats)
app.add_url_rule('/live_loans/', 'live_loans', live_loans)
app.add_url_rule('/model/', 'model', model)


def main():
    app.debug = False
    app.run()

if __name__ == "__main__":
    main()
