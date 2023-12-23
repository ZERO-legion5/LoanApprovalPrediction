from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():

    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def pred():

    no_of_dependents = request.form['no_of_dependents']
    education = request.form['education']
    self_employed = request.form['self_employed']
    income_annum = request.form['income_annum']
    loan_amount = request.form['loan_amount']
    loan_tenure = request.form['loan_tenure']
    cibil_score = request.form['cibil_score']
    residential_asset_value = request.form['residential_asset_value']
    commercial_asset_value = request.form['commercial_asset_value']
    luxury_asset_value = request.form['luxury_asset_value']
    bank_assets_value = request.form['bank_assets_value']
    
    m = [no_of_dependents,education,self_employed,income_annum,loan_amount,loan_tenure,cibil_score,residential_asset_value,commercial_asset_value,luxury_asset_value,bank_assets_value]
    data = np.array(list(map(int,m))).reshape(1,-1)
    df = pd.DataFrame(data,columns=['no_of_dependents', 'education', 'self_employed', 'income_annum',
       'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value',
       'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'])

    dt = joblib.load('models\\dtr.pkl')
    rf = joblib.load('models\\rfc.pkl')
    nb = joblib.load('models\\nb.pkl')
    xgb = joblib.load('models\\xgb.pkl')

    l = [dt, rf, nb, xgb]
    l = [x.predict(df) for x in l]

    m = []

    for i in l:

        m.append('Approved' if i == 1 else 'Not Approved')

    k = [3,4,2,5]

    print('DT :', m[0])
    print('RF :', m[1])
    print('NB :', m[2])
    print('XGB :', m[3])

    s = 0

    for i in range(len(l)):
        s += k[i] * l[i]

    if s > 7:

        m.append('Approved')

    else:

        m.append('Not Approved')

    return render_template('prediction.html', data=m)

if __name__ == '__main__':

    app.run(debug=True)