from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)
 
car = pd.read_csv('Cleaned_car.csv')
 
model = pickle.load(open('LinearRegression.pkl', 'rb'))
 
def get_dropdown_data():
    return {
        'companies': sorted(car['company'].unique()),
        'car_models': sorted(car['name'].unique()),
        'years': sorted(car['year'].unique(), reverse=True),
        'fuel_types': sorted(car['fuel_type'].unique())
    }

@app.route('/', methods=['GET'])
def index():
    dropdowns = get_dropdown_data()
    return render_template(
        'index.html',
        prediction=None, 
        selected_company='',
        selected_name='',
        selected_year='',
        selected_fuel_type='',
        selected_kms_driven='',
        **dropdowns
    )
@app.route('/predict', methods=['POST'])
def predict():
    dropdowns = get_dropdown_data()
    name = request.form.get('name', '')
    company = request.form.get('company', '')
    year = request.form.get('year', '')
    fuel_type = request.form.get('fuel_type', '')
    kms_driven = request.form.get('kms_driven', '')
 
    try:
        kms_driven_int = int(kms_driven)
        if kms_driven_int < 0:
            raise ValueError("Kilometers driven cannot be negative.")
    except:
        prediction = "Error: Kilometers driven must be a non-negative number."
        return render_template(
            'index.html',
            prediction=prediction,
            selected_company=company,
            selected_name=name,
            selected_year=year,
            selected_fuel_type=fuel_type,
            selected_kms_driven=kms_driven,
            **dropdowns
        )
 
    input_df = pd.DataFrame([[name, company, int(year), kms_driven_int, fuel_type]],
                            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
 
    prediction = round(model.predict(input_df)[0], 2)

    return render_template(
        'index.html',
        prediction=prediction, 
        selected_company=company,   
        selected_name=name,
        selected_year=year,
        selected_fuel_type=fuel_type,
        selected_kms_driven=kms_driven,
        **dropdowns
    )
if __name__ == '__main__':
    app.run(debug=True)
