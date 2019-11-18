import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go

fig = go.Figure()
def Tab_Scatter(df):
    return dcc.Tab(label = 'Trading Chart', value = 'tab-tiga', children = [
        html.Div(children = dcc.Graph(
        id = 'graph-scatter',
        figure = {
            'data' : [
                go.Scatter(
                x = df['Date'],
                y = df['Adj Close'],
                mode = 'lines',
                name = 'Price'),
                go.Scatter(
                x = df['Date'],
                y = df['BBLower'],
                mode = 'lines',
                name = 'Bollinger Band Lower'),
                 go.Scatter(
                x = df['Date'],
                y = df['SMA'],
                mode = 'lines',
                name = 'SMA')
                ],
            'layout' : go.Layout(
                xaxis = {'title' : 'Date'},
                yaxis = {'title' : 'Price'},
                hovermode = 'closest'
            )
        }
    )),
            html.Div(children = [
            dcc.Graph(
            id = 'Volume',
            figure = {
                'data' : [
                    {'x' : df['Date'], 'y' : df['Volume'], 'type' : 'bar', 'name' : 'Volume'} 
                ],
                'layout' : {'title' : 'Volume'}
            })])])

