from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import sqlite3
import json
import os, sys
import pandas as pd
from werkzeug.utils import secure_filename
from pathlib import Path
root_path = Path(__file__).resolve().parents[1]
sys.path.append(str(root_path))
from utils import save_config_field,load_config_field, load_record_with_lowest_confidence
from src import create_project_from_scratch, handle_new_record, process_new_records, force_process

app = Flask(__name__, static_folder='static')
CORS(app)

# Serve the main index page
@app.route('/')
def index():
    print("Call index.html")
    return render_template('index.html')

# Fetch all performances for a specific project and return as JSON
@app.route('/<project_name>/get_all_data', methods=['POST'])
def get_all_data(project_name):
    try:
        df = pd.read_csv(f"projects/{project_name}/eval_metrics_df.csv")
        df = df.fillna(value="null")
        data = df.to_dict(orient='list')
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 404

# List all projects stored in the projects folder
@app.route('/projects', methods=['GET'])
def list_projects():
    projects_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    projects_dir = os.path.join(projects_dir,"projects")
    projects = [d for d in os.listdir(projects_dir) if os.path.isdir(os.path.join(projects_dir, d))]
    return jsonify(projects)

# Retrieve configuration parameters for a specific project
@app.route('/<project_name>/getParams', methods=['GET'])
def get_parameters(project_name):
    config_path = os.path.join('projects', project_name, 'config_project.json')
    if os.path.exists(config_path):
        with open(config_path) as config_file:
            config_data = json.load(config_file)
            return jsonify(config_data)
    else:
        return jsonify({"error": "Configuration file not found."}), 404

# Trigger model training for a specific project
@app.route('/<project_name>/trainModel', methods=['GET'])
def launch_model_training(project_name):
    force_process(project_name)
    return jsonify({"success":"Correctly train the model"})

# Fetch the ID of the last processed record for a specific project
@app.route('/<project_name>/config/last_processed_record_id',methods=['GET'])
def get_last_processed(project_name):
    manual_annotations = load_config_field(project_name,"manual_annotations") +1
    val = jsonify({"manual_annotations":manual_annotations})
    return val
    
# Get counts of records based on their processing status for a specific project
@app.route('/<project_name>/records/status_count', methods=['GET'])
def get_records_status_count(project_name):    
    processed_not_trained = load_config_field(project_name,"numRecordsToTrain")
    processed_and_trained = load_config_field(project_name,"totalTrain")
    unprocessed = load_config_field(project_name,"num_records") - processed_not_trained - processed_and_trained
    return jsonify({
        "unprocessed": unprocessed,
        "processed_not_trained": processed_not_trained,
        "processed_and_trained": processed_and_trained
    })

# Fetch a record with the lowest confidence score from a specific project 
@app.route('/<project_name>/record', methods=['GET'])
def get_record(project_name):
    record = load_record_with_lowest_confidence(project_name)
    if record:
        return jsonify({
            "id": record[0],
            "text": record[1],
            "manual_labels": record[2], 
            "predicted_labels": record[3],
            "confidence": record[4]
        })
    else:
        return jsonify({"error": "Record not found"}), 404

# Retrieve labels for a specific project
@app.route('/<project_name>/labels', methods=['GET'])
def get_labels(project_name):
    config_path = os.path.join('projects', project_name, 'config_project.json')
    if os.path.exists(config_path):
        with open(config_path) as config_file:
            config_data = json.load(config_file)
            labels = config_data.get('labels', [])
            return jsonify(labels)
    else:
        return jsonify({"error": "Configuration file not found."}), 404

# Retrieve label colors for a specific project
@app.route('/<project_name>/label_colors', methods=['GET'])
def get_label_colors(project_name):
    config_path = os.path.join('projects', project_name, 'config_project.json')
    if os.path.exists(config_path):
        with open(config_path) as config_file:
            config_data = json.load(config_file)
            label_colors = config_data.get('labelColors', [])
            return jsonify(label_colors)
    else:
        return jsonify({"error": "Configuration file not found."}), 404

# Get performance metrics for a specific project
@app.route('/<project_name>/performances', methods=['GET'])
def get_performances(project_name):
    config_path = os.path.join('projects', project_name, 'config_project.json')
    if os.path.exists(config_path):
        with open(config_path) as config_file:
            config_data = json.load(config_file)
            performances = config_data.get('performances', [])
            print("Found the performances: ",performances)
            return jsonify(performances)
    else:
        return jsonify({"error": "Configuration file not found."}), 404

# Update a configuration parameter for a specific project
@app.route('/<project_name>/post_parameter', methods=['POST'])
def update_config(project_name):
    parameter = request.form.get('parameter')
    if parameter is None:
        return jsonify({"error": "Parameter is required"}), 400

    # Handle file uploads
    if 'value' in request.files:
        file = request.files['value']
        if file:
            filename = secure_filename(file.filename)
            file.save(f"projects/{project_name}/eval_set.tsv")
            save_config_field(project_name,"evalSetPath",f"projects/{project_name}/eval_set.tsv")
            return jsonify({"success": True, "file_saved": filename})
    else:
        value = request.form.get('value')
        if value is not None:
            try:
                value = json.loads(value)  # Parse the JSON string
            except json.JSONDecodeError:
                return jsonify({"error": "Invalid JSON format"}), 400
            
            print("Receive config change request")
            save_config_field(project_name, parameter, value)
            return jsonify({"success": True})

    return jsonify({"error": "Value is required"}), 400

# Update label information for a record in a specific project
@app.route('/<project_name>/update_label', methods=['POST'])
def update_label(project_name):
    data = request.json
    record_id = data['record_id']
    new_label = data['new_label']
    handle_new_record(project_name,new_label,record_id)
    return jsonify({"success": True})

# Process a new record for a specific project
@app.route('/<project_name>/process_new_record', methods=['POST'])
def process_record(project_name):
    process_new_records(project_name)
    return jsonify({"success": True})

# Create a new project from uploaded files
@app.route('/create_project', methods=['POST'])
def create_project():
    print("On create project")
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_name = request.form['projectName']
    project_dir = os.path.join(BASE_DIR, "projects", project_name)
    print(f"Project dir: {project_dir}")

    # Ensure the project directory exists
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)
        print(f"Has created the project directory at {project_dir}")

    # Process the main project file
    file = request.files.get('fileUpload')
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(project_dir, 'original_dataset.tsv')
        file.save(file_path)
        print(f"Saving main file to {file_path}")

    # Process the evaluation file, if present
    eval_file = request.files.get('evalFile')
    if eval_file:
        eval_filename = secure_filename(eval_file.filename)
        eval_file_path = os.path.join(project_dir, 'eval_set.tsv')
        eval_file.save(eval_file_path)

    # Assuming the existence of these helper functions
    create_project_from_scratch(project_name, file_path, request.form['labels'].split(','))
    save_config_field(project_name, "evalPercentage", float(request.form.get("evalPercentage", 0)))
    training_steps = int(request.form.get("trainingSteps", 5))
    num_predictions = int(request.form.get("numPredictions", 50))
    model_name = request.form.get("modelName", "")
    evalStartAt = int(request.form.get("evalStartAt",60))
    save_config_field(project_name,"trainingSteps",training_steps)
    save_config_field(project_name,"numPredictions",num_predictions)
    save_config_field(project_name,"modelName",model_name.lower())
    save_config_field(project_name,"startFrom",evalStartAt)
    if(eval_file):
        save_config_field(project_name,"evalSetPath",f"projects/{project_name}/eval_set.tsv")
    return jsonify({"success": True})

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=False)
