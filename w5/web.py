from waitress import serve
from flask import Flask, request, jsonify
import pickle


def load_artifact(path):
    with open(path,'rb') as f:        
        con = pickle.load(f)    
    return con

app = Flask('hm_predict')

@app.route('/predict', methods=['POST'])
def main():

    print('Receving request...')
    request_data = request.get_json()

    model = load_artifact('model1.bin')
    dv = load_artifact('dv.bin')

    tran_df = dv.transform(request_data)
    pred = model.predict_proba(tran_df)[0][1]

    print('Done executing...')
    return jsonify(pred)

if __name__ == "__main__":
    serve(app=app,host='0.0.0.0',port=9696)
