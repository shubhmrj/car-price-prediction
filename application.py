import pickle
from flask import Flask, render_template, request

application = Flask(__name__)
app = application

standard = pickle.load(open("Models/preprocessor.pkl","rb"))
randomsearchcv = pickle.load(open("Models/randomsearchcv.pkl","rb"))

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=="POST":
        model=float(request.form.get('model'))
        vehicle_age = float(request.form.get('vehicle_age'))
        km_driven = float(request.form.get('km_driven'))
        seller_type = float(request.form.get('seller_type'))
        fuel_type = float(request.form.get('fuel_type'))
        transmission_type = float(request.form.get('transmission_type'))
        mileage = float(request.form.get('mileage'))
        engine = float(request.form.get('engine'))
        max_power = float(request.form.get('max_power'))
        seats = float(request.form.get('seats'))

        new_data_scaled=standard.transform([[model,vehicle_age,km_driven,seller_type,fuel_type,transmission_type,mileage,engine,max_power,seats]])
        result=randomsearchcv.predict(new_data_scaled)

        return render_template('home.html',results=result[0])

       
    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")