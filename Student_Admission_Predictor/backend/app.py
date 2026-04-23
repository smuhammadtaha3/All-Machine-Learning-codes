from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import numpy as np
import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__, template_folder='../frontend')
CORS(app) 

# Model load karein

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models", "model.pkl") # changed

if not os.path.exists(model_path):
    raise FileNotFoundError("model.pkl not found in backend folder.")

with open(model_path, "rb") as f:
    model = pickle.load(f)

# Polynomial mapping function
def map_feature_single(x1, x2):
    degree = 6
    out = []
    for i in range(1, degree + 1):
        for j in range(i + 1):
            out.append((x1**(i - j) * (x2**j)))
    return np.array(out)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    try:
        raw_x1 = float(data["exam1"])
        raw_x2 = float(data["exam2"])
        
        # Scaling logic for 40+ scores
        x1_scaled = (raw_x1 / 60.0) - 0.5 
        x2_scaled = (raw_x2 / 60.0) - 0.5
        
        # Prediction calculation
        x_mapped = map_feature_single(x1_scaled, x2_scaled)
        z = np.dot(x_mapped, model["weights"]) + model["bias"]
        prob = 1 / (1 + np.exp(-z)) 
        prediction = 1 if prob >= 0.5 else 0

        # --- DYNAMIC PLOT WITH BACKGROUND DATA ---
        plt.figure(figsize=(6, 5))
        
        # 1. Background Data Load aur Plot karein
        # Make sure ex2data2.txt is in the same folder or provide full path
        try:
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(BASE_DIR, "../model_training/ex2data2.txt")
            data_points = np.loadtxt(data_path, delimiter=',')
            X_bg, y_bg = data_points[:, :2], data_points[:, 2]
            pos = y_bg == 1
            neg = y_bg == 0
            plt.scatter(X_bg[pos, 0], X_bg[pos, 1], marker='+', c='black', label='y=1 (Admitted)')
            plt.scatter(X_bg[neg, 0], X_bg[neg, 1], marker='o', c='y', edgecolors='k', label='y=0 (Not Admitted)')
        except:
            print("Warning: Background data file not found.")

        # 2. Decision Boundary (Green Line)
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z_grid = np.zeros((len(u), len(v)))
        for i in range(len(u)):
            for j in range(len(v)):
                mapped = map_feature_single(u[i], v[j])
                z_grid[i,j] = np.dot(mapped, model["weights"]) + model["bias"]
        plt.contour(u, v, z_grid.T, levels=[0], colors="green")

        # 3. User Point (Bright Red highlighted)
        plt.scatter(x1_scaled, x2_scaled, c='red', s=200, edgecolors='white', linewidth=2, label="Your Input", zorder=5)
        
        plt.title(f"Live Position: {'Admitted' if prediction == 1 else 'Not Admitted'}")
        plt.xlabel("Microchip Test 1 (Scaled)")
        plt.ylabel("Microchip Test 2 (Scaled)")
        plt.legend(loc="upper right", fontsize='small')
        plt.grid(True, linestyle='--', alpha=0.3)

        # Save plot to memory instead of disk
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        # Convert image to Base64
        image_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()

        return jsonify({
            "prediction": int(prediction),
            "graph": image_base64
        })
                
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/stats", methods=["GET"])
def get_stats():
    try:
        acc_path = os.path.join(BASE_DIR, "metadata", "accuracy.txt")
        with open(acc_path, "r") as f:
            acc = "".join(filter(lambda x: x.isdigit() or x == '.', f.read()))
        return jsonify({
            "accuracy": acc, 
            "default_graph": "/static/boundary_plot.png"
        })
    except:
        return jsonify({"error": "Stats not found"}), 404

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == "__main__":
    app.run(debug=True)