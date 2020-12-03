import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('INDEX.html')

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    """
    This Method Predict the abstract summary for the input data.

    :return: Abstract summary.

    wrriten by : Purvansh Singh
    Version : 1.0
    """
    data = request.json['data']

    # Checking for GPU
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Loading all the preprocessing required for the pegasus-wikihow Pretrained model
    tokenizer = AutoTokenizer.from_pretrained("google/pegasus-wikihow")

    # Loading model
    model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-wikihow")

    # summarizing
    batch = tokenizer.prepare_seq2seq_batch(data, truncation=True, padding='longest').to(torch_device)
    translated = model.generate(**batch)
    result = tokenizer.batch_decode(translated, skip_special_tokens=True)
    print(result)

    return jsonify({"text": result})



if __name__ == "__main__":
    #app.run(host='0.0.0.0', port=port)
    app.run(host='127.0.0.1', port=5000, debug=True)