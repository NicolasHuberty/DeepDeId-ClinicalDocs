// Import necessary functions from other modules
import { updateCountsUI, updateLabelSelectorColors, updateLabelsMappingsUI,fetchDataAndUpdateAllPlots, updatePerformancesDisplay,selectProject } from './main.js';
import { getCurrentProject, setCurrentRecordId, setLabelColors,setAllManualLabels, getManualLabels,getAllManualLabels, getCurrentRecordId,
    setTrainingStep,incrementOnTraining,getOnTraining,decrementOnTraining } from './stateManager.js';

// Function to fetch and display projects
export function loadProjects() {
    fetch('/projects', { method: 'GET' })
        .then(response => response.json())
        .then(data => {
            const projectsList = document.getElementById('projectsList');
            projectsList.innerHTML = '';
            data.forEach(project => {
                const projectElement = document.createElement('div');
                projectElement.textContent = project;
                projectElement.classList.add('btn', 'btn-primary', 'm-2');
                projectElement.onclick = () => selectProject(project);
                projectsList.appendChild(projectElement);
            });
        })
        .catch(error => {
            console.error('Error loading projects:', error);
            alert('Error loading projects: ' + error.message);
        });
}

// Function to update the progress bars
export function loadRecordCounts() {
    fetch(`/${getCurrentProject()}/records/status_count`, { method: 'GET' })
        .then(response => response.json())
        .then(data => {
            if (data) {
                updateCountsUI(data.processed_not_trained, data.processed_and_trained, data.unprocessed);
            }
        })
        .catch(error => {
            console.error('Error loading record counts:', error);
            alert('Error loading record counts: ' + error.message);
        });
}

// Upload the labels from the project and update colors and keyboard mapping
export function loadLabels(projectName) {
    fetch(`/${projectName}/labels`, { method: 'GET' })
        .then(response => response.json())
        .then(data => {
            if (!data.error) {
                const labelsList = document.getElementById('labelSelector');
                while (labelsList.options.length > 0) { 
                    labelsList.remove(0);
                }
                data.forEach(label => {
                    const option = document.createElement('option');
                    option.value = label;
                    option.textContent = label;
                    labelsList.appendChild(option);
                });
                updateLabelSelectorColors();
                updateLabelsMappingsUI();
            } else {
                console.error('Error loading labels');
                alert('Error loading labels');
            }
        })
        .catch(error => {
            console.error('Error loading labels:', error);
            alert('Error loading labels: ' + error.message);
        });
}

// Function to fetch and display project parameters
export function fetchParameters(projectName) {
    return fetch(`/${projectName}/getParams`, { method: 'GET' })
        .then(response => response.json())
        .then(data => {
            document.getElementById('evalPercentage').value = data.evalPercentage;
            document.getElementById('startFrom').value = data.startFrom;
            document.getElementById('trainingSteps').value = data.trainingSteps;
            document.getElementById('numPred').value = data.numPredictions;
            if(data.performances){
                updatePerformancesDisplay(data.performances);
            }
            setLabelColors(data.labelColors);
            updateLabelSelectorColors();
            updateLabelsMappingsUI();
            setTrainingStep(data.trainingSteps);
            return data;
        })
        .catch(error => {
            console.error('Error fetching project data:', error);
            alert('Error fetching project data: ' + error.message);
        });
}

// Function to train the model
export function trainModel(projectName){
    incrementOnTraining();
    const trainingProgressText = document.getElementById('trainingProgressText');
    trainingProgressText.textContent = "The model is currently being trained...";
    document.getElementById('trainingProgressBar').style.width = '100%';
    document.getElementById('trainingProgressBar').classList.add('progress-bar-striped', 'progress-bar-animated');
    return fetch(`/${projectName}/trainModel`, { method: 'GET' })
    .then(data => {
        decrementOnTraining();
        const progressBar = document.getElementById('trainingProgressBar');
        progressBar.classList.remove('progress-bar-striped', 'progress-bar-animated');
        loadRecordCounts();
    })
    .catch(error => {
        console.error('Error training the model:', error);
        alert('Error training the model: ' + error.message);
        decrementOnTraining();
    });
}

// Load and display a record
export function loadRecord() {
    fetch(`/${getCurrentProject()}/record`, { method: 'GET' })
        .then(response => response.json())
        .then(data => {
            if (!data.error) {
                const text = data.text;
                setAllManualLabels(data.predicted_labels);
                const tokens = text;
                let coloredText = '';
                tokens.forEach((token, index) => {
                    const label = getManualLabels(index); 
                    let spanStyle = '';
                    let spanClass = 'token-label';
                    let labelName = '';

                    if (label !== 'O') {
                        spanStyle = `background-color: #008000; color: #ffffff;`;
                        labelName = `<div style="color: #008000; font-size: small;">${label}</div>`;
                    } else {
                        spanStyle = 'background-color: transparent;';
                    }
                    coloredText += `<div style="display: inline-block; text-align: center;">${labelName}<span class="${spanClass}" data-index="${index}" style="${spanStyle}">${token}</span></div> `;
                });
                document.getElementById('recordText').innerHTML = coloredText;
                setCurrentRecordId(data.id);
            } else {
                console.error('Record not found');
                alert('Record not found');
            }
        })
        .catch(error => {
            console.error('Error loading record:', error);
            alert('Error loading record: ' + error.message);
        });
}

// Create a new project
export function createProject() {
    $('#loadingText').show();
    const projectName = document.getElementById('projectNameCreation').value;
    const labels = document.getElementById('projectLabelsCreation').value.replace(/\s+/g, '');
    const fileInput = document.getElementById('projectFileCreation');
    const evalFile = document.getElementById('evaluationFileCreation');
    const evalPercentage = document.getElementById("evalPercentageCreation").value;
    const evalStartAt = document.getElementById("evalStartAtCreation").value;
    const trainingSteps = document.getElementById('trainingStepsCreation').value;
    const numPredictions = document.getElementById('numPredictionsCreation').value;
    const modelName = document.getElementById('modelNameCreation').value;

    const formData = new FormData();
    formData.append('projectName', projectName);
    formData.append('labels', labels);
    formData.append('fileUpload', fileInput.files[0]);
    formData.append("evalFile", evalFile.files[0]);
    formData.append("evalPercentage", evalPercentage);
    formData.append("evalStartAt", evalStartAt);
    formData.append('trainingSteps', trainingSteps);
    formData.append('numPredictions', numPredictions);
    formData.append("modelName", modelName);

    fetch('/create_project', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert('Project created successfully!');
            loadProjects();
            $('#createProjectModal').modal('hide');
            $('#loadingText').hide();
        } else {
            alert('Error creating project');
        }
    })
    .catch(error => {
        console.error('Error creating project:', error);
        alert('Error creating project: ' + error.message);
    });
}

// Submit annotations done
export function submitAnnotations() {
    return new Promise((resolve, reject) => {
        const joinedLabels = getAllManualLabels().join(",");
        fetch(`/${getCurrentProject()}/update_label`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                record_id: getCurrentRecordId(),
                new_label: joinedLabels
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                loadRecord();
                resolve(); 
                fetchParameters(getCurrentProject());
            } else {
                alert('Error updating annotations');
                reject('Error updating annotations');
            }
        })
        .catch(error => {
            console.error('Error updating annotations:', error);
            alert('Error updating annotations: ' + error.message);
            reject(error);
        });
    });
}

// Process a record annotated - Handle the training progress of the model
export function processNewRecord() {
    const onTraining = getOnTraining();
    if (onTraining !== 0) {
        console.log("Training in progress... skipping new process");
        return;
    }
    incrementOnTraining(); 
    const trainingProgressText = document.getElementById('trainingProgressText');

    const progressBar = document.getElementById('trainingProgressBar');

    const loadingTimeout = setTimeout(() => {
        trainingProgressText.textContent = "The model is currently being trained...";
        document.getElementById('trainingProgressBar').style.width = '100%';
        document.getElementById('trainingProgressBar').classList.add('progress-bar-striped', 'progress-bar-animated');
    }, 2500);

    fetch(`/${getCurrentProject()}/process_new_record`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'}
    })
    .then(response => response.json())
    .then(data => {
        console.log("Processing completed");
        decrementOnTraining();
        if (getOnTraining() === 0) {
            clearTimeout(loadingTimeout);
            progressBar.classList.remove('progress-bar-striped', 'progress-bar-animated');
            if (data.success) {
                loadRecordCounts();
                fetchDataAndUpdateAllPlots();
            } else {
                alert('Error processing new record');
            }
        }
    })
    .catch(error => {
        console.error('Error processing new record:', error);
        alert('Error processing new record: ' + error.message);
        decrementOnTraining();
    });
}

// Function to post a configuration change
export function postNewConfig(parameter, value) {
    const url = `/${getCurrentProject()}/post_parameter`;

    const formData = new FormData();
    formData.append('parameter', parameter);
    
    if (value instanceof File) {
        formData.append('value', value);
    } else {
        formData.append('value', JSON.stringify(value));
    }

    fetch(url, {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        return data;
    })
    .catch(error => {
        console.error('Error posting new configuration:', error);
        alert('Error posting new configuration: ' + error.message);
    });
}