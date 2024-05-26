let state = {
    currentProject: '',
    currentRecordId: 0,
    manual_labels: [],
    labelColors: {},
    trainingStep: 5,
    onTraining: 0
};

export const getState = () => state;

export const setState = (newState) => {
    state = { ...state, ...newState };
};

export const setLabelColors = (labelColors) => {
    state.labelColors = labelColors;
};

export const setCurrentProject = (currentProject) => {
    state.currentProject = currentProject;
}

export const setCurrentRecordId = (currentRecordId) => {
    state.currentRecordId = currentRecordId;
}

export const setManualLabels = (manualLabels,index) => {
    state.manual_labels[index] = manualLabels;
}
export const setAllManualLabels = (manualLabels) => {
    state.manual_labels = manualLabels;
}
export const setTrainingStep = (trainingStep) => {
    state.trainingStep = trainingStep;
}
export const incrementOnTraining = () => {
    state.onTraining = state.onTraining + 1;
}
export const decrementOnTraining = () => {
    state.onTraining = state.onTraining - 1;
}
export const getLabelColors = (label) => state.labelColors[label];
export const getAllLabelColors = () => state.labelColors;
export const getCurrentProject = () => state.currentProject;

export const getCurrentRecordId = () => state.currentRecordId;

export const getManualLabels = (index) => state.manual_labels[index];
export const getAllManualLabels = ()=> state.manual_labels;
export const getTrainingSteps = () => state.trainingStep;
export const getOnTraining = () => state.onTraining;