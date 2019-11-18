import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html

def Tab_DataTable(df):
    return dcc.Tab(label = 'DataFrame Table', value = 'tab-satu', children = [
        html.Div(id = 'data-table', children = dash_table.DataTable(
            id = 'Table',
            columns = [{'name' : i, 'id' : i} for i in df.columns],
            data = df.to_dict('records'),
            page_action = 'native',
            page_current = 0,
            page_size = 10
            ))       
    ])  

        