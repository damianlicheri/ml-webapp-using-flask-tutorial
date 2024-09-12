from flask import Flask, request, render_template
from pickle import load
import pandas as pd

app = Flask(__name__)
model = load(open("../models/random_forest_regressor_42.sav", "rb"))
#scaler = load(open("../models/scaler.sav", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    try:
        if request.method == "POST":
            val1 = float(request.form["CRIM"])
            val2 = float(request.form["RM"])
            val3 = float(request.form["DIS"])
            val4 = float(request.form["LSTAT"])
            
            # Crear un array con los valores
            data = [[val1, val2, val3, val4]]
            
            # Convertir el array a un DataFrame con nombres de columnas
            data_df = pd.DataFrame(data, columns=['CRIM', 'RM', 'DIS', 'LSTAT'])
            
            # Escalar los datos
            #data_scaled = scaler.transform(data_df)
            
            # Realizar la predicción
            prediction = model.predict(data_df)[0]
            pred_class = f"{prediction:.3f} M$"
        else:
            pred_class = None
        return render_template("index.html", prediction=pred_class)
    except Exception as e:
        return str(e)

# Habilitar el modo de depuración y ejecutar la aplicación
if __name__ == "__main__":
    app.debug = True
    app.run()