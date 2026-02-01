from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/default')
def default_page():
    return render_template('default.html')

@app.route('/dashboard')
def dashboard():
    df = pd.read_csv('loan_approval_dataset.csv')
    df.columns = df.columns.str.strip()
    
    # Summary Stats
    stats = {
        'total_loans': len(df),
        'approved_count': len(df[df['loan_status'] == ' Approved']),
        'rejected_count': len(df[df['loan_status'] == ' Rejected']),
        'avg_income': round(df['income_annum'].mean(), 2),
        'avg_loan': round(df['loan_amount'].mean(), 2),
        'avg_cibil': round(df['cibil_score'].mean(), 1)
    }
    
    # Education breakdown
    edu_stats = df.groupby('education')['loan_status'].value_counts().unstack().fillna(0).to_dict('index')
    
    # CIBIL range distribution
    bins = [300, 450, 600, 750, 900]
    labels = ['300-450', '450-600', '600-750', '750-900']
    df['cibil_range'] = pd.cut(df['cibil_score'], bins=bins, labels=labels)
    cibil_dist = df['cibil_range'].value_counts().sort_index().to_dict()

    return render_template('dashboard.html', stats=stats, edu_stats=edu_stats, cibil_dist=cibil_dist)

@app.route('/predict', methods=['POST'])
def pred():
    try:
        data_dict = request.form.to_dict()
        m = [
            data_dict['no_of_dependents'],
            data_dict['education'],
            data_dict['self_employed'],
            data_dict['income_annum'],
            data_dict['loan_amount'],
            data_dict['loan_tenure'],
            data_dict['cibil_score'],
            data_dict['residential_asset_value'],
            data_dict['commercial_asset_value'],
            data_dict['luxury_asset_value'],
            data_dict['bank_assets_value']
        ]
        
        data = np.array(list(map(int, m))).reshape(1, -1)
        df = pd.DataFrame(data, columns=['no_of_dependents', 'education', 'self_employed', 'income_annum',
                                       'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value',
                                       'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'])

        dt = joblib.load('models/dtr.pkl')
        rf = joblib.load('models/rfc.pkl')
        nb = joblib.load('models/nb.pkl')
        xgb = joblib.load('models/xgb.pkl')

        l = [dt, rf, nb, xgb]
        predictions = [int(x.predict(df)[0]) for x in l]
        
        # Calculate probabilities for risk scoring (using RFC and XGB as primary indicators)
        try:
            rf_prob = rf.predict_proba(df)[0][1] # Probability of 'Approved'
            xgb_prob = xgb.predict_proba(df)[0][1]
            avg_prob = (rf_prob + xgb_prob) / 2
            
            # For Approval Predictor: Success Score
            success_score = round(avg_prob * 100, 1)
            
            # For Default Predictor: Probability of Default (PD)
            # PD is high when Approval Prob is low. 
            # We can also weigh in CIBIL score and Debt-to-Income ratio for a more realistic PD
            pd_score = round((1 - avg_prob) * 100, 1)
        except:
            success_score = (sum(predictions) / len(predictions)) * 100
            pd_score = 100 - success_score

        results = ['Approved' if p == 1 else 'Not Approved' for p in predictions]

        k = [3, 4, 2, 5]
        s = sum(k[i] * predictions[i] for i in range(len(predictions)))
        
        final_status = 'Approved' if s > 7 else 'Not Approved'
        results.append(final_status)

        return jsonify({
            'individual_results': {
                'dt': results[0],
                'rf': results[1],
                'nb': results[2],
                'xgb': results[3]
            },
            'final_result': results[4],
            'success_score': success_score,
            'default_probability': pd_score
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
