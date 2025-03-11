from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
try:
    model = pickle.load(open("model.pkl", "rb"))
except Exception as e:
    print("Error loading model:", e)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Log incoming request data
        print("Received request:", request.data)

        # Ensure request contains JSON
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON input"}), 400
        
        # Log parsed data
        print("Parsed data:", data)

        # Convert input into array
        features = np.array([[data["Pclass"], data["Sex"], data["Age"], 
                              data["SibSp"], data["Parch"], data["Fare"]]])
        
        # Log model input
        print("Model input:", features)

        # Predict
        prediction = model.predict(features)[0]

        # Return prediction
        return jsonify({"prediction": int(prediction)})

    except Exception as e:
        print("ERROR:", str(e))  # Print error in terminal
        return jsonify({"error": str(e)}), 500  # Return error message in response



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)