// Import necessary functions from other modules
import { getLabelColors, getCurrentProject,setCurrentProject, setManualLabels, getTrainingSteps, setTrainingStep,getAllLabelColors } from './stateManager.js';
import {loadProjects,loadRecordCounts,loadLabels,loadRecord,createProject,submitAnnotations,fetchParameters,processNewRecord,postNewConfig,trainModel} from './api.js';



// Update the UI counts for processed and unprocessed records
export function updateCountsUI(processedNotTrained, processedAndTrained, unprocessed) {
    const total = processedNotTrained + processedAndTrained + unprocessed;
    const processedPercentage = (((processedNotTrained + processedAndTrained) / total) * 100).toFixed(2);
    const unprocessedPercentage = ((unprocessed / total) * 100).toFixed(2);
    // Set the text content for the counts and percentages
    document.getElementById('processedCount').textContent = processedNotTrained + processedAndTrained;
    document.getElementById('unprocessedCount').textContent = unprocessed;
    document.getElementById('processedPercentage').textContent = `${processedPercentage}%`;
    document.getElementById('unprocessedPercentage').textContent = `${unprocessedPercentage}%`;
    
    // Set the widths of the progress bars
    const processedBar = document.getElementById('processedBar');
    const unprocessedBar = document.getElementById('unprocessedBar');
    
    if (processedBar) {
        processedBar.style.width = `${processedPercentage}%`;
        processedBar.classList.add('bg-warning', 'bg-success');
    }
    if (unprocessedBar) {
        unprocessedBar.style.width = `${unprocessedPercentage}%`;
        unprocessedBar.classList.add('bg-danger');
    }
    // Set the training progress bar and text
    const trainingGoal = getTrainingSteps(); 
    const trainingProgress = Math.min(processedNotTrained, trainingGoal);
    const trainingProgressPercentage = ((trainingProgress / trainingGoal) * 100).toFixed(2);
    document.getElementById('trainingProgressBar').style.width = `${trainingProgressPercentage}%`;
    document.getElementById('trainingProgressText').textContent = `Training Progress: ${trainingProgress}/${trainingGoal}`;
}

// Update the label selector colors
export function updateLabelSelectorColors() {
    const labelsList = document.getElementById('labelSelector');
    if (labelsList) {
        for (let i = 0; i < labelsList.options.length; i++) {
            const option = labelsList.options[i];
            const color = getLabelColors(option.value);
            if (color) {
                option.style.backgroundColor = color;
                option.style.color = '#ffffff';
            }
        }
    }
}

// Handle token selection for manual annotation
export function selectToken(index) {
    const selector = document.getElementById('labelSelector');
    const tokenElement = document.querySelector(`[data-index="${index}"]`);
    const rect = tokenElement.getBoundingClientRect();
    selector.style.display = 'block';
    selector.style.position = 'absolute';
    selector.style.left = `${rect.left}px`;
    selector.style.top = `${rect.bottom + window.scrollY}px`;
    selector.setAttribute('data-selected-index', index);
}

// Handle project selection
export function selectProject(projectName) {
    setCurrentProject(projectName);
    loadRecord(); 
    createLabelSelector();
    loadLabels(projectName);
    loadRecordCounts();
    fetchParameters(projectName);
    fetchDataAndUpdateAllPlots();
    loadRecord();
    document.getElementById("mainContent").classList.remove("hidden");
}

// Set the label for a selected token
function setLabelForToken(label) {
    const index = document.getElementById('labelSelector').getAttribute('data-selected-index');
    if (index !== null) {
        const tokenElement = document.querySelector(`[data-index="${index}"]`);
        const parentDiv = tokenElement.parentElement;
        if (parentDiv) {
            // Remove the label name div above the word
            const labelNameDiv = parentDiv.querySelector('div');
            if (labelNameDiv) {
                labelNameDiv.remove();
            }
        }
        // Set the background color of the token
        tokenElement.style.backgroundColor = getLabelColors(label);
        tokenElement.style.color = '#000000';  // Set text color to black if needed
        setManualLabels(label, index);
    }
    document.getElementById('labelSelector').style.display = 'none';
    const selector = document.getElementById('labelSelector');
    // Reset selector
    if (selector) {
        selector.value = "";
    }
}

// Create the label selector UI
function createLabelSelector() {
    const existingSelector = document.getElementById('labelSelector');
    if (!existingSelector) {
        const selector = document.createElement('select');
        selector.id = 'labelSelector';
        selector.onchange = function() { setLabelForToken(this.value); };
        const defaultOption = document.createElement('option');
        defaultOption.value = "";
        defaultOption.textContent = "Select a label";
        defaultOption.disabled = true;
        defaultOption.selected = true;
        selector.appendChild(defaultOption);

        selector.style.display = 'none'; // Initially hidden
        document.body.appendChild(selector); 
    }
}

// Update the label mappings UI
export function updateLabelsMappingsUI() {
    const labelsMappingsContainer = document.getElementById('labelsMappingsContainer');
    labelsMappingsContainer.innerHTML = ''; // Clear existing content
    const options = document.getElementById('labelSelector').options;
    for (let i = 0; i < options.length; i++) {
        const option = options[i];
        const labelText = option.text;
        
        // Create a label element with consistent styling
        const labelElement = document.createElement('div');
        labelElement.classList.add('label-mapping');
        labelElement.textContent = `${labelText}: ${i}`;
        labelElement.style.display = 'inline-block';
        labelElement.style.margin = '10px 5px';
        labelElement.style.padding = '5px';
        labelElement.style.borderRadius = '10px';
        labelElement.style.backgroundColor = getLabelColors(labelText);
        if (labelText == 'O') {
            labelElement.style.border = "2px solid black";
            labelElement.style.color = "black";
        }

        labelsMappingsContainer.appendChild(labelElement);
    }
}

// Fetch data and update all plots
export function fetchDataAndUpdateAllPlots() {
    const projectName = getCurrentProject();
    fetch(`/${projectName}/get_all_data`, { method: 'POST' }) // Using POST for potential future use
        .then(response => {
            if (response.ok) {
                return response.json();
            } else {
                return;
            }
        })
        .then(data => {
            updatePlot('plotF1Scores', data, Object.keys(getAllLabelColors()), 'Labels F1-Score', 'F1-Score');
            updatePlot('plotMacroAvg', data, ['macro avg'], 'Average F-1 Score');
            updatePlot('plotConfidence', data, ['confidence'], 'Confidence');
        })
        .catch(error => {
            console.error('Error fetching and updating plots:', error);
        });
}

// Update the performance display
export function updatePerformancesDisplay(performances) {
    const performancesContainer = document.getElementById('performancesDisplay');
    performancesContainer.innerHTML = ''; // Clear previous entries
    // Display only individual label supports
    Object.entries(performances).forEach(([label, stats]) => {
        if (!['macro avg', 'micro avg', 'weighted avg'].includes(label) && stats.support !== undefined) {
            const supportElement = document.createElement('p');
            const supportText = typeof stats.support === 'number' ? stats.support.toLocaleString() : 'Data not available';
            supportElement.textContent = `${label} Support: ${supportText}`;
            performancesContainer.appendChild(supportElement);
        }
    });
}

// Update a specific plot
function updatePlot(plotId, data, labels, title, yAxisTitle = 'Values') {
    const labelColors = getAllLabelColors();
    const traces = labels.map(label => ({
        x: data['Trained Records Size'],
        y: data[label],
        type: 'scatter',
        mode: 'lines+markers',
        name: label,
        marker: { color: label === 'O' ? '#000000' : labelColors[label] }
    }));

    const layout = {
        title: title,
        xaxis: { title: 'Trained Records Size' },
        yaxis: { title: yAxisTitle }
    };

    Plotly.newPlot(plotId, traces, layout);
}

// Initial call to populate the project list
document.addEventListener('DOMContentLoaded', function() {
    loadProjects();
});

// Display selected file name
document.getElementById('projectFileCreation').addEventListener('change', function() {
    const fileName = this.files[0].name;
    const nextSibling = this.nextElementSibling;
    nextSibling.innerText = fileName;
});

// Handle submit annotations button click
document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('submitAnnotationsButton').addEventListener('click', async function() {
        try {
            await submitAnnotations();
            processNewRecord();
        } catch (error) {
            console.error('Error submitting annotations:', error);
            alert('Error submitting annotations: ' + error.message);
        }
    });
});

// Handle create project button click
document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('createProject').addEventListener('click', createProject);
});

// Handle token selection for manual annotation
document.addEventListener('DOMContentLoaded', function() {
    const recordTextElement = document.getElementById('recordText');
    recordTextElement.addEventListener('click', function(event) {
        if (event.target.classList.contains('token-label')) {
            const index = event.target.getAttribute('data-index');
            if (index !== null) {
                selectToken(index);
            }
        }
    });
});

// Handle Enter key to submit annotations
document.addEventListener('keydown', function(event) {
    if (event.key === 'Enter') {
        submitAnnotations().catch(error => {
            console.error('Error submitting annotations:', error);
            alert('Error submitting annotations: ' + error.message);
        });
    }
});

// Handle label selection with number keys
document.addEventListener('keydown', function(event) {
    const selector = document.getElementById('labelSelector');
    if (selector && selector.style.display !== 'none') {
        const key = parseInt(event.key, 10);
        const labels = selector.getElementsByTagName('option');
        if (!isNaN(key) && key >= 0 && key < labels.length) {
            selector.value = labels[key].value;
            setLabelForToken(labels[key].value);
            event.preventDefault();
        }
    }
});

// Handle input element changes and save configuration
const inputElements = [
    { id: 'evalPercentage', type: 'number' },
    { id: 'startFrom', type: 'number' },
    { id: 'trainingSteps', type: 'number' },
    { id: 'numPred', type: 'number' },
    { id: 'evaluationFile', type: 'file' }
];

inputElements.forEach(input => {
    const element = document.getElementById(input.id);
    if (element) {
        element.addEventListener('change', function(event) {
            let value;
            if (input.type === 'number') {
                value = parseInt(element.value);
                if (input.id === 'trainingSteps') {
                    setTrainingStep(value);
                    loadRecordCounts();
                }
                if (isNaN(value)) {
                    console.error(`Invalid input: ${input.id} is not a number`);
                    alert(`Invalid input: ${input.id} is not a number`);
                    return;
                }
            } else if (input.type === 'file') {
                value = element.files[0];
            } else {
                value = element.value;
            }
            postNewConfig(input.id, value)
                .then(() => {
                    alert(`${element.labels[0].textContent} has been correctly saved!`);
                })
                .catch(error => {
                    console.error(`Error saving ${input.id}:`, error);
                    alert(`Error saving ${input.id}: ${error.message}`);
                });
        });
    }
});

// Handle force training button click
document.getElementById("forceTraining").addEventListener("click", function(event) {
    trainModel(getCurrentProject()).catch(error => {
        console.error('Error training model:', error);
        alert('Error training model: ' + error.message);
    });
});