
# Based on https://github.com/allenai/allennlp-server/blob/master/allennlp_server/commands/server_simple.py

import argparse
import json
import logging
import os
import torch
import youtokentome

import nltk
nltk.download('punkt')
from nltk import tokenize

from flask import Flask, request, Response, jsonify
from flask import render_template
from flask_cors import CORS
from gevent.pywsgi import WSGIServer

from models import *
from inference import inference
from utils import seed_everything

logger = logging.getLogger(__name__)

def make_app(model, tokenizer, title) -> Flask:
    app = Flask(__name__, template_folder=os.path.dirname(os.path.realpath(__file__)))

    @app.route("/")
    def index() -> Response:
        models = ['Extractive Model']
        return render_template("template.html", title=title, models=models)


    @app.route("/predict", methods=["POST", "OPTIONS"])
    def predict() -> Response:
        if request.method == "OPTIONS":
            return Response(response="", status=200)

        data = request.get_json()   
        
        seed_everything()
        max_sentences = 30
        max_sentence_length = 30
        lower = False
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sentences = [sentence.lower() if lower else sentence for sentence in tokenize.sent_tokenize(data['source'])][:max_sentences]

        inputs = [tokenizer.encode(sentence)[:max_sentence_length] for sentence in sentences]

        max_sentence_length = max(max_sentence_length, max([len(tokens) for tokens in inputs]))
        tensor_inputs = torch.zeros((max_sentences, max_sentence_length), dtype=torch.long, device=device)
        
        for i, inputs in enumerate(inputs):
            for j, sentence_tokens in enumerate(inputs):
                tensor_inputs[i][j] = torch.tensor(sentence_tokens, dtype=torch.int64)

        txt = {'inputs': tensor_inputs.unsqueeze(0),
              'records': sentences
              }
        prediction = inference(model, txt, device, top_k=3, threshold=0)
        
        return jsonify(prediction)

    return app

def main(model_dir, tokenizer_dir,  title, host, port): #, include_package):
  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = youtokentome.BPE(tokenizer_dir)
    vocabulary = tokenizer.vocab()
    vocab_size = len(vocabulary)
    model = SentenceTaggerRNN(vocab_size).to(device)
    checkpoint = torch.load(model_dir, map_location='cpu') 


    model.load_state_dict(checkpoint['state_dict'])
    app = make_app(
        model,
        tokenizer,
        title=title,
    )
    CORS(app)

    http_server = WSGIServer((host, port), app)
    print(f"Models loaded, serving demo on http://{host}:{port}")
    http_server.serve_forever()

if __name__ == "__main__":
   
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./model.pt", help="directory with all model archives") #required=True, 
    parser.add_argument('--tokenizer_dir', type=str, default="./BPE_model.bin",  help='BPE tokenizer directory') 
    parser.add_argument("--title", type=str, help="change the default page title", default="Extractive Text Summarization")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="interface to serve the demo on")
    parser.add_argument("--port", type=int, default=5000, help="port to serve the demo on")
   
    args = parser.parse_args()
    main(**vars(args))
