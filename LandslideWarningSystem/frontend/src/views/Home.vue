<template>
  <div class="dashboard-container">
    <!-- Header -->
    <header class="header">
      <div class="logo">Landslide Warning System</div>
      <div class="user-info">
        <span>Welcome, Admin</span>
        <el-button type="text" @click="logout">Logout</el-button>
      </div>
    </header>

    <div class="main-content">
      <!-- Sidebar -->
      <aside class="sidebar">
        <el-menu default-active="1" class="el-menu-vertical">
          <el-menu-item index="1">
            <el-icon><Monitor /></el-icon>
            <span>Dashboard</span>
          </el-menu-item>
          <el-menu-item index="2">
            <el-icon><Document /></el-icon>
            <span>Data Management</span>
          </el-menu-item>
          <el-menu-item index="3">
            <el-icon><Setting /></el-icon>
            <span>Settings</span>
          </el-menu-item>
        </el-menu>
      </aside>

      <!-- Content Area -->
      <main class="content">
        <div class="map-container" id="map"></div>
        
        <!-- Control Panel -->
        <div class="control-panel">
          <el-card class="box-card">
            <template #header>
              <div class="card-header">
                <span>Analysis Control</span>
              </div>
            </template>
            <div class="actions">
              <el-upload
                class="upload-demo"
                action="#"
                :auto-upload="false"
                :on-change="handleFileChange"
                :show-file-list="false"
              >
                <el-button type="primary">Select Image</el-button>
              </el-upload>
              <div v-if="selectedFile" class="file-info">
                Selected: {{ selectedFile.name }}
                <el-button type="success" size="small" @click="startAnalysis" :loading="loading">
                  Start Analysis
                </el-button>
              </div>
            </div>
            
            <div v-if="taskStatus" class="status-info">
              <p>Status: <el-tag :type="statusType">{{ taskStatus }}</el-tag></p>
              <p v-if="landslideCount !== null">Detected: {{ landslideCount }} landslides</p>
            </div>
          </el-card>
        </div>
      </main>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';
import { useRouter } from 'vue-router';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import api from '../services/api';
import { ElMessage } from 'element-plus';
import { Monitor, Document, Setting } from '@element-plus/icons-vue';

const router = useRouter();
const map = ref(null);
const selectedFile = ref(null);
const loading = ref(false);
const taskStatus = ref(null);
const landslideCount = ref(null);
let pollInterval = null;

// Initialize Map
onMounted(() => {
  map.value = L.map('map').setView([30.0, 104.0], 13); // Default view (China approximate)
  
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors'
  }).addTo(map.value);
});

const logout = () => {
  router.push('/login');
};

const handleFileChange = (file) => {
  selectedFile.value = file.raw;
  taskStatus.value = null;
  landslideCount.value = null;
};

const statusType = computed(() => {
  if (taskStatus.value === 'completed') return 'success';
  if (taskStatus.value === 'failed') return 'danger';
  if (taskStatus.value === 'processing') return 'warning';
  return 'info';
});

import { computed } from 'vue';

const startAnalysis = async () => {
  if (!selectedFile.value) return;
  
  loading.value = true;
  taskStatus.value = 'uploading';
  
  try {
    // 1. Upload
    const uploadRes = await api.uploadImage(selectedFile.value);
    const imageId = uploadRes.data.image_id;
    
    // 2. Predict
    const predictRes = await api.predict(imageId);
    const taskId = predictRes.data.task_id;
    
    taskStatus.value = 'processing';
    
    // 3. Poll Status
    pollInterval = setInterval(async () => {
      const statusRes = await api.getTaskStatus(taskId);
      const status = statusRes.data.status;
      taskStatus.value = status;
      
      if (status === 'completed' || status === 'failed') {
        clearInterval(pollInterval);
        loading.value = false;
        
        if (status === 'completed') {
          landslideCount.value = statusRes.data.landslides_count;
          loadResults(taskId);
          ElMessage.success('Analysis completed successfully!');
        } else {
          ElMessage.error('Analysis failed.');
        }
      }
    }, 2000);
    
  } catch (error) {
    console.error(error);
    loading.value = false;
    taskStatus.value = 'error';
    ElMessage.error('An error occurred during analysis.');
  }
};

const loadResults = async (taskId) => {
  try {
    const res = await api.getResults(taskId);
    const geojson = res.data;
    
    if (geojson.features.length > 0) {
      // Add GeoJSON to map
      const layer = L.geoJSON(geojson, {
        style: {
          color: '#ff0000',
          weight: 2,
          opacity: 1,
          fillOpacity: 0.5
        },
        onEachFeature: (feature, layer) => {
          layer.bindPopup(`Confidence: ${feature.properties.confidence}`);
        }
      }).addTo(map.value);
      
      map.value.fitBounds(layer.getBounds());
    }
  } catch (error) {
    console.error("Error loading results:", error);
  }
};
</script>

<style scoped>
.dashboard-container {
  height: 100vh;
  display: flex;
  flex-direction: column;
}

.header {
  height: 60px;
  background-color: #001529;
  color: white;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 20px;
}

.logo {
  font-size: 20px;
  font-weight: bold;
}

.user-info span {
  margin-right: 15px;
}

.main-content {
  flex: 1;
  display: flex;
  overflow: hidden;
}

.sidebar {
  width: 200px;
  background-color: white;
  border-right: 1px solid #e6e6e6;
}

.el-menu-vertical {
  height: 100%;
  border-right: none;
}

.content {
  flex: 1;
  position: relative;
}

.map-container {
  width: 100%;
  height: 100%;
  z-index: 1;
}

.control-panel {
  position: absolute;
  top: 20px;
  right: 20px;
  width: 300px;
  z-index: 1000;
}

.file-info {
  margin-top: 10px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.status-info {
  margin-top: 15px;
  border-top: 1px solid #eee;
  padding-top: 10px;
}
</style>
