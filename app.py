import os
import subprocess
import time
from threading import Thread
from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, Response
)
import pandas as pd
import joblib
from werkzeug.utils import secure_filename
from graphs import generate_graphs

# ─── Flask configuration ────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Secret key for session management
app.config['UPLOAD_FOLDER'] = 'data/'
ALLOWED_EXTENSIONS = {'csv'}

# ─── Load model & dropdown data ─────────────────────────────────────────────────
model         = joblib.load('model/model.pkl')
features      = joblib.load('model/features.pkl')
dropdown_data = joblib.load('model/dropdown_data.pkl')

# ─── Global training progress tracker ───────────────────────────────────────────
training_progress = {'percent': 0}

def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    )

# ─── Background training function ───────────────────────────────────────────────
def train_background():
    training_progress['percent'] = 0  # Reset progress

    proc = subprocess.Popen(
        ['python', 'train_model.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in proc.stdout:
        print(line.strip())  # Log output
        if line.startswith('PROGRESS:'):
            try:
                progress = int(line.strip().split(':')[1])
                training_progress['percent'] = progress
                print(f"Progress updated: {progress}%")  # Debug line
            except ValueError:
                pass

    proc.wait()
    training_progress['percent'] = 100  # Complete
    print("Training complete.")  # Debug line

# ─── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def root():
    return redirect(url_for('dashboard'))

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if request.method == 'POST':
        # Handle file upload
        if 'csv_file' in request.files and request.files['csv_file']:
            file = request.files['csv_file']
            if file and allowed_file(file.filename):
                filename  = secure_filename(file.filename)
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(save_path)
                flash(f"📁 Uploaded '{filename}' successfully.", 'success')
            else:
                flash("❗ Please upload a valid .csv file.", 'danger')
                return redirect(url_for('dashboard'))

        # Handle training trigger
        if request.form.get('action') == 'train':
            Thread(target=train_background, daemon=True).start()
            flash("🚀 Training started! Watch progress below.", 'success')

    return render_template('dashboard.html', dropdown_data=dropdown_data)

@app.route('/train_progress')
def train_progress():
    def event_stream():
        last_sent = -1
        while True:
            progress = training_progress.get('percent', 0)
            if progress != last_sent:
                last_sent = progress
                yield f"data:{progress}\n\n"
            if last_sent >= 100:
                break
            time.sleep(0.2)
    return Response(event_stream(), mimetype='text/event-stream')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        input_data = request.form.to_dict()
        df_input = pd.DataFrame([input_data])
        df_input['Year']  = int(df_input['Year'])
        df_input['Month'] = int(df_input['Month'])
        df_input['Day']   = int(df_input['Day'])

        # One-hot encode the data
        df_encoded = pd.get_dummies(df_input)
        for col in features:
                if col not in df_encoded.columns:
                    df_encoded[col] = 0  # Add the missing column with 0 value
        df_encoded = df_encoded[features]  # Ensure column alignment

        # Predict
        prediction = model.predict(df_encoded)[0]

        # Load dataset
        df = pd.read_csv('data/ev_sales_india.csv')
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['EV_Sales_Quantity'] = df['EV_Sales_Quantity'].fillna(df['EV_Sales_Quantity'].median())
        df['State'] = df['State'].str.strip()

        df_filtered = df[
            (df['Vehicle_Class'] == input_data['Vehicle_Class']) &
            (df['Vehicle_Category'] == input_data['Vehicle_Category']) &
            (df['Vehicle_Type'] == input_data['Vehicle_Type'])
        ].copy()

        if df_filtered.empty:
            df_filtered = df_input.copy()
            df_filtered['EV_Sales_Quantity'] = prediction
            df_filtered['Date'] = pd.to_datetime(df_input[['Year','Month','Day']])

        if 'Month' not in df_filtered.columns:
            df_filtered['Month'] = df_filtered['Date'].dt.month

        # Generate graphs
        generate_graphs(df_filtered)
        graph_paths = {
            'ev_sales_trend': 'graphs/ev_sales_trend.png',
            'ev_sales_by_state': 'graphs/ev_sales_by_state.png',
            'vehicle_type_distribution': 'graphs/vehicle_type_distribution.png',
            'state_month_heatmap': 'graphs/state_month_heatmap.png'
        }

        return render_template(
            'result.html',
            prediction=round(prediction, 2),
            graph_paths=graph_paths
        )

    return render_template(
        'index.html',
        states=dropdown_data['State'],
        vehicle_classes=dropdown_data['Vehicle_Class'],
        vehicle_categories=dropdown_data['Vehicle_Category'],
        vehicle_types=dropdown_data['Vehicle_Type']
    )

@app.route('/index')
def index():
    return render_template('index.html')

# ─── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True)
