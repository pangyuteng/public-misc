from plotly.offline import plot
from plotly.graph_objs import Scatter
from flask import Flask, render_template, request, jsonify


app = Flask(__name__,
    static_url_path='', 
    static_folder='static',
    template_folder='templates',
)

@app.route("/ping")
def ping():
    return jsonify(success=True)

@app.route('/finance/us-market-overview')
def us_market_overview():
    x_data = [0,1,2,3]
    y_data = [x**2 for x in x_data]
    plot_div = plot([
        Scatter(x=x_data, y=y_data,
            mode='lines', name='test',
            opacity=0.8, marker_color='green')
        ],output_type='div',include_plotlyjs=False))
    return render_template("main.html", plot_div=plot_div)

if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0")