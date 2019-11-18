import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import DataTable
import Scatter
from dash.dependencies import Input, Output, State
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,roc_auc_score, f1_score, log_loss, matthews_corrcoef
from sklearn.model_selection import train_test_split
pd.set_option('display.max_columns', None)
from xgboost import XGBClassifier
from yahoo_historical import Fetcher

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets = external_stylesheets)
lq45 = ['ADHI.JK','ADRO.JK','ASII.JK','ASRI.JK','BBCA.JK','BBNI.JK','BBRI.JK','BBTN.JK','BKSL.JK','BMRI.JK',
       'CPIN.JK','INTP.JK','ITMG.JK','JSMR.JK','KLBF.JK','LPKR.JK','LPPF.JK','MNCN.JK','PTBA.JK','PTPP.JK','SCMA.JK','SMGR.JK',
       'SSMS.JK','TPIA.JK','UNTR.JK','UNVR.JK','WIKA.JK','WSKT.JK','ELSA.JK','EXCL.JK','GGRM.JK','HMSP.JK','ICBP.JK','INCO.JK','INDF.JK','INDY.JK']
dfstock = input('Stocks: ')
def bestresults(stock):
    accuracies = []
    for n in range(1,31):
        df = Fetcher(stock,[2010,1,1],[2019,1,1]).getHistorical()
        df.drop(['Open','Close'], axis = 1, inplace = True)
        df['Date'] = pd.to_datetime(df['Date'])
        df.replace(0, np.nan, inplace=True)
        df.dropna(inplace = True)
        ma = []
        for i in range(len(df)):
            if i < 14:
                ma.append(np.nan)
            else:
                ma.append(df['Adj Close'][(i-14):i].mean())
        df['SMA'] = ma
        BBLower = []
        for i in range(len(df)):
            if i < 14 :
                BBLower.append(np.nan)
            else:
                BBLower.append((df['SMA'].iloc[i] - (2 * df['Adj Close'][(i-14):i].std())))
        df['BBLower'] = BBLower
        williamsr= []
        for i in range(len(df)):
            if i < 14 :
                williamsr.append(np.nan)
            else:
                high = df['High'].iloc[(i-14):(i)].max()
                low = df['Low'].iloc[(i-14):(i)].min()
                williamsr.append(((high-df['Adj Close'].iloc[i])/(high-low))*(-100))
        df['Williams R'] = williamsr
        result = []
        for i in range(len(df)):
            if i < n:
                result.append(np.nan)
            elif i > (len(df)-(n+1)):
                result.append(np.nan)
            else:
                if df['Adj Close'].iloc[i] < df['Adj Close'].iloc[i+n]:
                    result.append('Profit')
                else:
                    result.append('Loss')
        df['Result'] = result
        df.dropna(axis = 0, inplace = True)
        data = df.drop(['Date','Result','Adj Close', 'High','Low'],axis = 1)
        result = df['Result']
        X_train, X_test, y_train, y_test = train_test_split(data, result, test_size = 0.2)
        X_train, X_test, y_train, y_test = train_test_split(data, result, test_size = 0.2)
        xgboost = XGBClassifier(learning_rate = 0.01, n_estimators = 1000, max_depth = 4)
        xgboost.fit(X_train, y_train)
        param_learning = (0.12, 0.1, 0.08, 0.06)
        param_estimator = (80,100,120,140)
        max_depth = (1,2,3,4)
        param_grid = {'learning_rate': param_learning, 'n_estimators':param_estimator, 'max_depth':max_depth}
        gs = GridSearchCV(xgboost,param_grid,scoring='accuracy')
        gs = gs.fit(X_train, y_train)
        learning_rates = 0
        n_estimatorss = 0
        max_depths = 0
        learning_rates += gs.best_params_['learning_rate']
        n_estimatorss += gs.best_params_['n_estimators']
        max_depths += gs.best_params_['max_depth']
        xgboost = XGBClassifier(learning_rate = learning_rates, n_estimators = n_estimatorss, max_depth = max_depths)
        xgboost.fit(X_train, y_train)
        predictions = xgboost.predict(X_test)
        accuracies.append(accuracy_score(y_test,predictions))
    angka = pd.DataFrame(accuracies)
    bestresult = (angka[0].sort_values(ascending = False).reset_index()['index'][0] + 1)
    return bestresult
df = Fetcher(dfstock,[2010,1,1],[2019,1,1]).getHistorical()
df.drop(['Open','Close'], axis = 1, inplace = True)
df['Date'] = pd.to_datetime(df['Date'])
df.replace(0, np.nan, inplace=True)
df.dropna(inplace = True)
ma = []
for i in range(len(df)):
    if i < 14:
        ma.append(np.nan)
    else:
        ma.append(df['Adj Close'][(i-14):i].mean())
df['SMA'] = ma
BBLower = []
for i in range(len(df)):
    if i < 14 :
        BBLower.append(np.nan)
    else:
        BBLower.append((df['SMA'].iloc[i] - (2 * df['Adj Close'][(i-14):i].std())))
df['BBLower'] = BBLower
williamsr= []
for i in range(len(df)):
    if i < 14 :
        williamsr.append(np.nan)
    else:
        high = df['High'].iloc[(i-14):(i)].max()
        low = df['Low'].iloc[(i-14):(i)].min()
        williamsr.append(((high-df['Adj Close'].iloc[i])/(high-low))*(-100))
df['Williams R'] = williamsr
result = []
for i in range(len(df)):
    if i < 14:
        result.append(np.nan)
    elif i > (len(df)-15):
        result.append(np.nan)
    else:
        if df['Adj Close'].iloc[i] < df['Adj Close'].iloc[i+14]:
            result.append('Profit')
        else:
            result.append('Loss')
df['Result'] = result
df.dropna(axis = 0, inplace = True)
def model(stock):
    df = Fetcher(stock,[2010,1,1],[2019,1,1]).getHistorical()
    df.drop(['Open','Close'], axis = 1, inplace = True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.replace(0, np.nan, inplace=True)
    df.dropna(inplace = True)
    ma = []
    for i in range(len(df)):
        if i < 14:
            ma.append(np.nan)
        else:
            ma.append(df['Adj Close'][(i-14):i].mean())
    df['SMA'] = ma
    BBLower = []
    for i in range(len(df)):
        if i < 14 :
            BBLower.append(np.nan)
        else:
            BBLower.append((df['SMA'].iloc[i] - (2 * df['Adj Close'][(i-14):i].std())))
    df['BBLower'] = BBLower
    williamsr= []
    for i in range(len(df)):
        if i < 14 :
            williamsr.append(np.nan)
        else:
            high = df['High'].iloc[(i-14):(i)].max()
            low = df['Low'].iloc[(i-14):(i)].min()
            williamsr.append(((high-df['Adj Close'].iloc[i])/(high-low))*(-100))
    df['Williams R'] = williamsr
    result = []
    for i in range(len(df)):
        if i < 14:
            result.append(np.nan)
        elif i > (len(df)-15):
            result.append(np.nan)
        else:
            if df['Adj Close'].iloc[i] < df['Adj Close'].iloc[i+14]:
                result.append('Profit')
            else:
                result.append('Loss')
    df['Result'] = result
    df.dropna(axis = 0, inplace = True)
    data = df.drop(['Date','Result','Adj Close', 'High','Low'],axis = 1)
    result = df['Result']
    X_train, X_test, y_train, y_test = train_test_split(data, result, test_size = 0.2)
    X_train, X_test, y_train, y_test = train_test_split(data, result, test_size = 0.2)
    xgboost = XGBClassifier(learning_rate = 0.01, n_estimators = 1000, max_depth = 4)
    xgboost.fit(X_train, y_train)
    param_learning = (0.12, 0.1, 0.08, 0.06)
    param_estimator = (80,100,120,140)
    max_depth = (1,2,3,4)
    param_grid = {'learning_rate': param_learning, 'n_estimators':param_estimator, 'max_depth':max_depth}
    gs = GridSearchCV(xgboost,param_grid,scoring='accuracy')
    gs = gs.fit(X_train, y_train)
    learning_rates = 0
    n_estimatorss = 0
    max_depths = 0
    learning_rates += gs.best_params_['learning_rate']
    n_estimatorss += gs.best_params_['n_estimators']
    max_depths += gs.best_params_['max_depth']
    xgboost = XGBClassifier(learning_rate = learning_rates, n_estimators = n_estimatorss, max_depth = max_depths)
    xgboost.fit(X_train, y_train)
    predictions = xgboost.predict(X_test)
    angka = 0
    angka += bestresults(stock) 
    predictions = xgboost.predict(X_test)
    resultnew = []
    for i in range(len(df)):
        if i <  angka:
            resultnew.append(np.nan)
        elif i > (len(df)-(angka + 1)):
            resultnew.append(np.nan)
        else:
            if df['Adj Close'].iloc[i] < df['Adj Close'].iloc[i+angka]:
                resultnew.append('Profit')
            else:
                resultnew.append('Loss')
    df['Result'] = resultnew
    xgboost = XGBClassifier(learning_rate = learning_rates, n_estimators = n_estimatorss, max_depth = max_depths)
    xgboost.fit(X_train, y_train)
    dfpredict = Fetcher(stock,[2019,10,8],[2019,11,18]).getHistorical()
    dfpredict = dfpredict[['Volume','Adj Close']]
    dfpredict['SMA'] = dfpredict['Adj Close'][-angka:-1].mean()
    dfpredict['BBLower'] = df['SMA'].iloc[-1] - (2 * df['Adj Close'][-angka:].std())
    dfpredict['Williams R'] = ((high-df['Adj Close'].iloc[-angka])/(high-low))*(-100)
    dfpredict.drop('Adj Close', axis =1, inplace = True)
    df2 = []
    if xgboost.predict(pd.DataFrame(dfpredict.iloc[-1]).transpose())[0] == 'Loss':
          df2.append([stock,
                xgboost.predict(pd.DataFrame(dfpredict.iloc[-1]).transpose())[0],
                pd.DataFrame(xgboost.predict_proba(pd.DataFrame(dfpredict.iloc[-1]).transpose()))[0][0],
                accuracy_score(y_test,predictions),angka])
    elif xgboost.predict(pd.DataFrame(dfpredict.iloc[-1]).transpose())[0] == 'Profit':
          df2.append([stock,
                xgboost.predict(pd.DataFrame(dfpredict.iloc[-1]).transpose())[0],
                pd.DataFrame(xgboost.predict_proba(pd.DataFrame(dfpredict.iloc[-1]).transpose()))[1][0],
                accuracy_score(y_test,predictions),angka])
    return df2
df3 = []
df3.append(model(dfstock)[0])
dffinal =pd.DataFrame(df3, columns = ['Stocks', 'Prediction', 'Probability', 'Accuracy','Best Time'])

app.layout = html.Div(children = [
    html.Center(html.H1('Stock Prediction')),
    html.P('Created by: Lazuardi'),
    dcc.Tabs(value = 'tabs', id = 'tabs-1', children = [
        DataTable.Tab_DataTable(dffinal), 
        Scatter.Tab_Scatter(df)
        ],
    content_style = {
        'fontFamily' : 'Arial',
        'borderBottom' : '1px solid #d6d6d6',
        'borderLeft' : '1px solid #d6d6d6',
        'borderRight' : '1px solid #d6d6d6',
        'padding' : '44px'
        })
    ],
    style = {
        'maxWidth' : '1200px',
        'margin' : '0 auto'
    })


if __name__ == '__main__':
    app.run_server(debug=True)