import os
import traceback
import datetime
import pandas as pd

from plotly.offline import plot
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from flask import Flask, render_template, request, jsonify

from utils import get_data, sector_list

app = Flask(__name__,
    static_url_path='', 
    static_folder='static',
    template_folder='templates',
)

@app.route("/ping")
def ping():
    return jsonify(success=True)

@app.route('/finance/spy')
def spy():
    
    tstamp =  datetime.datetime.now().strftime("%Y-%m-%d")
    cache_csv = f'{tstamp}.csv'

    lookback = -2000
    roll = 200
    if not os.path.exists(cache_csv):
        df = get_data(lookback=lookback,roll=roll)
        df.to_csv(cache_csv,index=True)

    df = pd.read_csv(cache_csv)

    last_date = df.Date.tolist()[-1]
    start_date = df.Date.tolist()[-1]

    spy_plot_div = plot([
        go.Scatter(
            x=df['Date'][lookback:],
            y=df['SPY'][lookback:],
            mode='lines', name='SPY',
            opacity=0.8, marker_color='blue')
        ],output_type='div',include_plotlyjs=False)

    vix_plot_div = plot([
        go.Scatter(
            x=df['Date'][lookback:],
            y=df['^VIX'][lookback:],
            mode='lines', name='^VIX',
            opacity=0.8, marker_color='blue')
        ],output_type='div',include_plotlyjs=False)

    return render_template(
        "spy.html",
        spy_plot_div=spy_plot_div,
        vix_plot_div=vix_plot_div,
    )


#@app.route('/finance/us-market-overview')
@app.route('/')
def us_market_overview():
    try:
        lookback = int(request.args.get("lookback",0))
        if lookback > 0:
            lookback = -1*lookback
        roll = int(request.args.get("roll",200))

        tstamp =  datetime.datetime.now().strftime("%Y-%m-%d")
        cache_csv = f'{tstamp}-lookback{lookback}-roll{roll}.csv'
        if not os.path.exists(cache_csv):
            df = get_data(lookback=lookback,roll=roll)
            df.to_csv(cache_csv,index=True)

        df = pd.read_csv(cache_csv)

        last_date = df.Date.tolist()[-1]
        start_date = df.Date.tolist()[-1]

        btcusd = go.Scatter(
            x=df['Date'][lookback:],
            y=df['BTC-USD'][lookback:],
            mode='lines', name='BTC-USD',
            opacity=0.8, marker_color='orange')

        m2 = go.Scatter(
            x=df['Date'][lookback:],
            y=df['M2SL'][lookback:],
            mode='lines', name='M2SL',
            opacity=0.8, marker_color='green')

        spy = go.Scatter(
            x=df['Date'][lookback:],
            y=df['SPY'][lookback:],
            mode='lines', name='SPY',
            opacity=0.8, marker_color='blue')

        vix = go.Scatter(
            x=df['Date'][lookback:],
            y=df['^VIX'][lookback:],
            mode='lines', name='^VIX',
            opacity=0.8, marker_color='purple')

        yield_diff = go.Scatter(
            x=df['Date'][lookback:],
            y=(df['^TNX']-df['^IRX'])[lookback:],
            mode='lines', name='^TNX-^IRX',
            opacity=0.8, marker_color='red')

        corr_to_spy_list = []
        for x in sector_list:
            item = go.Scatter(
                x=df['Date'][lookback:],
                y=df[f'{x}_corr'][lookback:],
                mode='lines', name=f'corr({x},SPY)',
                opacity=0.8)
            corr_to_spy_list.append(item)

        fig = make_subplots(
            rows=6, cols=1, shared_xaxes=True, 
            vertical_spacing=0.02
        )

        fig.add_trace(btcusd,row=1, col=1)

        fig.add_trace(m2,row=2, col=1)

        fig.add_trace(spy,row=3, col=1)

        fig.add_trace(vix,row=4, col=1)

        fig.add_trace(yield_diff,row=5, col=1)

        for x in corr_to_spy_list:
            fig.add_trace(x,row=6, col=1)

        fig.update_layout(height=600, width=1000,
                    title_text="market overview")

        fig['layout']['yaxis1'].update(domain=[0, 0.5])
        fig['layout']['yaxis2'].update(domain=[0.5, 0.6])
        fig['layout']['yaxis3'].update(domain=[0.6, 0.7])
        fig['layout']['yaxis4'].update(domain=[0.7, .8])
        fig['layout']['yaxis5'].update(domain=[0.8,.95])
        fig['layout']['yaxis6'].update(domain=[0.95, 1])
        fig.update_layout(template='plotly_dark')

        plot_div = plot(fig,output_type='div',include_plotlyjs=False)
        
        return render_template(
            "main.html",
            plot_div=plot_div,
            last_updated_tstamp=tstamp,
        )
    except:
        traceback.print_exc()
        return jsonify({"message":traceback.format_exc()})

if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0")