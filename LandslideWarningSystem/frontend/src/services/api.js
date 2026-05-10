import axios from 'axios';

const apiClient = axios.create({
  baseURL: '/api/analysis',
  headers: {
    'Content-Type': 'application/json',
  },
});

export default {
  uploadImage(file) {
    const formData = new FormData();
    formData.append('file', file);
    return apiClient.post('/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },

  predict(imageId) {
    return apiClient.post('/predict', { image_id: imageId });
  },

  getTaskStatus(taskId) {
    return apiClient.get(`/tasks/${taskId}`);
  },

  getResults(taskId) {
    return apiClient.get(`/results/${taskId}`);
  },
};
