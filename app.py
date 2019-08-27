import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import base64
import os
from urllib.parse import quote as urlquote
from flask import Flask, send_from_directory
from keras.preprocessing import image
from keras.models import load_model
from keras import backend as K
import numpy as np
from os import sys
import tensorflow as tf

UPLOAD_DIRECTORY = "upload"

import time

server = Flask(__name__)
app = dash.Dash(server=server)

app.config['suppress_callback_exceptions']=True


app.layout = html.Div([
    
	html.Div([
		html.Div([
			html.H3("Aivoverenvuodon ja Keuhkokuumeen tunnistus röntgenkuvasta konvoluutioneuroverkolla")
			], className="banner", style={"margin-top": "10px", "margin-bottom": "20px"}),

		html.Div([
				dcc.Tabs(id="tabs",
					children=[
				        dcc.Tab(label='Aivoverenvuoto', value='av'),
				        dcc.Tab(label='Keuhkokuume', value='kk'),
    				]
    			),
		], className="container"),

		html.Div(id="tab-output"),

	], className="ten columns offset-by-one", style={"backgroundColor": "#f0f5f5", "height":"95vh", "border-radius": "40px", "margin-top": "20px"}),
			html.Div([
			dcc.Markdown("""
_Tensorflow version: 1.14.0,_
_Keras Version: 2.2.4_
""")  
		], className="ten columns offset-by-one", style = {"height":"5vh"}),

], className="row", style={"textAlign": "center", "backgroundColor": "#d1e0e0", "height":"105vh"})

av_layout = [
	html.Div([
		html.H1("Aivoverenvuoto", style = {"margin-top": "20ox"}),
		    dcc.Upload(
		        id="upload-image",
		        children=html.Div([
		            "Drag and Drop or ",
		            html.A("Select File")
		        ]),
		        style={
		            "lineHeight": "60px",
		            "borderWidth": "1px",
		            "borderStyle": "dashed",
		            "borderRadius": "5px",
		            "textAlign": "center",
		            "margin": "10px",
		            "fontSize": 20,
		            "position": "center"
		        },
		        multiple=True
		    ),
		dcc.Loading(id="loading-1", children=[html.Div(id="loading-output-1")], type="default"),
	    ], className="container"),
		html.Div(id="output-image-upload"),
		html.H2(id="text"),
]

kk_layout = [
	html.Div([
		html.H1("Keuhkokuume", style = {"margin-top": "20ox"}),
		    dcc.Upload(
		        id="upload-image",
		        children=html.Div([
		            "Drag and Drop or ",
		            html.A("Select File")
		        ]),
		        style={
		            "lineHeight": "60px",
		            "borderWidth": "1px",
		            "borderStyle": "dashed",
		            "borderRadius": "5px",
		            "textAlign": "center",
		            "margin": "10px",
		            "fontSize": 20,
		            "position": "center"
		        },
		        multiple=True
		    ),
		dcc.Loading(id="loading-1", children=[html.Div(id="loading-output-1")], type="default"),
	    ], className="container"),
		
		html.Div(id="output-image-upload"),
		html.H2(id="text"),
]


@server.route("/download/<path:path>")
def download(path):
    """Serve a file from the upload directory."""
    return send_from_directory(UPLOAD_DIRECTORY, path, as_attachment=True)


def save_file(name, content):
    """Decode and store a file uploaded with Plotly Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join(UPLOAD_DIRECTORY, name), "wb") as fp:
        fp.write(base64.decodebytes(data))

def remove_pic(name):
	os.remove("upload" + "/" + "{0}".format(name))


def prediction(name, model):
	time.sleep(2)
	file = "upload/" + "{0}".format(name)
	
	if model == "av":
		img = image.load_img(file, target_size = (224, 224))
		img = image.img_to_array(img)
		img = np.expand_dims(img, axis = 0)
		img /= 255.
		model = load_model("av_model2.h5")
		prediction = model.predict(img)
		K.clear_session()
		remove_pic(name)

		if prediction[0] > 0.5:
			return("Ei aivoverenvuotoa {:.2%} varmuudella".format(prediction[0][0]))
		else:
			return("Aivoverenvuoto {:.2%} varmuudella".format(1-prediction[0][0]))

	else:
		img = image.load_img(file, target_size = (224, 224))
		img = image.img_to_array(img)
		img = np.expand_dims(img, axis = 0)
		img /= 255.
		model = load_model("kk_model.h5")

		prediction = model.predict(img)
		K.clear_session()
		remove_pic(name)

		if prediction[0] > 0.5:
			return ("Keuhkokuume {:.2%} varmuudella".format(prediction[0][0]))
		else:
			return ("Ei keuhkokuumetta {:.2%} varmuudella".format(1-prediction[0][0]))

def parse_contents(name, content):
    return html.Div([
        html.H5(name),
        html.Img(src=content, style={"with": "450px", "height": "450px"}),
]),

@app.callback(
	Output('tab-output', 'children'),
	[Input('tabs', 'value')])

def show_content(value):
	if value == "av":
		return av_layout
	if value == "kk":
		return kk_layout
	else:
		return None


@app.callback([
	Output("loading-output-1", "children"),
	Output("output-image-upload", "children"),
	Output("text", "children")],
	[Input("tabs", "value"),
	Input("upload-image", "contents"),
	Input("upload-image", "filename")])

def update_output(model, list_of_names, list_of_contents):
	if list_of_contents is not None:
		for name, data in zip(list_of_contents, list_of_names):
			save_file(name, data)
		time.sleep(1)
		return None, parse_contents(name, data), prediction(name, model)


if __name__ == "__main__":
    app.run_server(debug=True, port=9980)