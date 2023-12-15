

import joblib
from flask import Flask, request, jsonify
import pandas as pd
 
model=joblib.load(r"model.pkl")
bp_encoder=joblib.load(r"bp_encoder.pkl")
gen_encoder=joblib.load(r"gen_encoder.pkl")
drug_encoder=joblib.load(r"drug_encoder.pkl")
cho_encoder=joblib.load(r"cho_encoder.pkl")
 
app = Flask(__name__)
 
@app.route('/',methods=['POST'])
def predict():
    data=request.get_json(force=True)
    data=data['test']
    print(data)
    df=pd.DataFrame([data])
    print(df)
    df['Sex']=gen_encoder.transform(df['Sex'])
    df['BP']=bp_encoder.transform(df['BP'])
    df['Cholesterol']=cho_encoder.transform(df['Cholesterol'])
    out=model.predict(df)
    print(out)
    output=drug_encoder.inverse_transform(out)
    return str(output)

app.run(host='0.0.0.0')