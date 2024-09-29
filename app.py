from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from joblib import load
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
import uuid

app = Flask(__name__)


@app.route('/', methods=['GET', "POST"])
def hello_world():  # put application's code here
    request_type_str = request.method
    if request_type_str == "GET":
        return render_template('index.html', href='static/base_pic.svg')
    else:
        text = request.form['text']
        random_string = uuid.uuid4().hex
        path = "static/" + random_string + ".svg"
        model = load("model.joblib")
        np_arr = floats_str_to_np_arr(text)
        make_picture('AgesAndHeights.pkl', model, np_arr, path)

        return render_template('index.html', href=path)


def make_picture(training_data_filename, model, new_inp_np_arr, output_file):
    # read in the data
    data = pd.read_pickle(training_data_filename)
    age = data["Age"]
    age = data[age > 0]
    height = data["Height"]

    x_new = np.array(list(range(19))).reshape(19, 1)

    preds = model.predict(x_new)

    fig = px.scatter(x=age, y=height, title="Height vs age of people",
                     labels={'x': 'Age in years', 'y': 'Height in inches'})

    fig.add_trace(go.Scatter(x=x_new.reshape(19), y=preds, mode='lines', name="model"))

    new_preds = model.predict(new_inp_np_arr)

    fig.add_trace(
        go.Scatter(x=new_inp_np_arr.reshape(len(new_inp_np_arr)), y=new_preds, name="new_output", mode='markers',
                   marker=dict(color='purple', size=20, line=dict(color='purple', width=2))))

    # Save the image as a .svg file
    fig.write_image(output_file, width=500, engine='kaleido')

    fig.show()


def floats_str_to_np_arr(floats_str):
    def is_float(s):
        try:
            float(s)
            return True
        except:
            return False

    floats = np.array([float(x) for x in floats_str.split(",") if is_float(x)])
    return floats.reshape(len(floats), 1)


if __name__ == '__main__':
    app.run()
