from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    if request.method == "POST":
        hours = float(request.form["hours"])
        pred = model.predict([[hours]])
        prediction = f"Predicted Marks: {pred[0]:.2f}"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)