<template>
  <div class="dashboard-container">
    <!-- Header -->
    <header class="header">
      <div class="header-left">
        <div class="logo">油气管道滑坡智能预警系统</div>
      </div>
      <div class="header-center">
        <el-menu
          mode="horizontal"
          background-color="#001529"
          text-color="#fff"
          active-text-color="#409EFF"
          :ellipsis="false"
          router
        >
          <el-menu-item index="/">监测总览</el-menu-item>
          <el-menu-item index="/data-management">数据管理</el-menu-item>
          <el-menu-item index="/sample-augmentation">样本增强</el-menu-item>
          <el-menu-item index="/system-settings">系统设置</el-menu-item>
        </el-menu>
      </div>
      <div class="user-info">
        <span>欢迎, 管理员</span>
        <el-button type="text" @click="logout">退出登录</el-button>
      </div>
    </header>

    <div class="main-content">
      <!-- Full Screen Map -->
      <div class="map-container" id="map"></div>

      <!-- Left Panel -->
      <div class="panel left-panel">
        <el-card class="panel-card">
          <template #header>
            <div class="card-header">
              <span>数据统计</span>
            </div>
          </template>
          <div class="stat-grid">
            <div class="stat-item">
              <div class="stat-value">1,205</div>
              <div class="stat-label">接入影像 (幅)</div>
            </div>
            <div class="stat-item">
              <div class="stat-value">3,450</div>
              <div class="stat-label">管道里程 (km)</div>
            </div>
          </div>
        </el-card>

        <el-card class="panel-card">
          <template #header>
            <div class="card-header">
              <span>预警概览</span>
            </div>
          </template>
          <div id="pie-chart" class="chart-container"></div>
        </el-card>
      </div>

      <!-- Right Panel -->
      <div class="panel right-panel">
        <el-card class="panel-card">
          <template #header>
            <div class="card-header">
              <span>实时告警列表</span>
            </div>
          </template>
          <div class="alert-list">
            <div
              v-for="(alert, index) in alerts"
              :key="index"
              class="alert-item"
              @click="viewAlert(alert)"
            >
              <el-tag :type="getAlertTagType(alert.level)" size="small">{{
                alert.levelText
              }}</el-tag>
              <span class="alert-time">{{ alert.time }}</span>
              <span class="alert-loc">{{ alert.location }}</span>
            </div>
          </div>
        </el-card>

        <el-card class="panel-card">
          <template #header>
            <div class="card-header">
              <span>滑坡面积排行 (Top 5)</span>
            </div>
          </template>
          <div id="bar-chart" class="chart-container"></div>
        </el-card>
      </div>

      <!-- Operation Console (Floating) -->
      <div class="control-panel">
        <el-card class="box-card">
          <template #header>
            <div class="card-header">
              <span>智能分析控制台</span>
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
              <el-button type="primary" size="small">上传影像并分析</el-button>
            </el-upload>
            <div v-if="selectedFile" class="file-info">
              <span class="file-name">{{ selectedFile.name }}</span>
              <el-button
                type="success"
                size="small"
                @click="startAnalysis"
                :loading="loading"
              >
                开始
              </el-button>
            </div>
          </div>
          <div v-if="taskStatus" class="status-info">
            <p>
              状态:
              <el-tag :type="statusType" size="small">{{
                getStatusText(taskStatus)
              }}</el-tag>
            </p>
            <p v-if="landslideCount !== null">
              结果: {{ landslideCount }} 处风险
            </p>
          </div>
        </el-card>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, computed, nextTick } from "vue";
import { useRouter } from "vue-router";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import * as echarts from "echarts";
import api from "../services/api";
import { ElMessage } from "element-plus";

const router = useRouter();
const map = ref(null);
const selectedFile = ref(null);
const loading = ref(false);
const taskStatus = ref(null);
const landslideCount = ref(null);
let pollInterval = null;

// Mock Data for Alerts
const alerts = ref([
  { level: "high", levelText: "高风险", time: "10:23", location: "K120+500" },
  { level: "medium", levelText: "中风险", time: "09:45", location: "K115+200" },
  { level: "low", levelText: "低风险", time: "08:30", location: "K098+100" },
  { level: "high", levelText: "高风险", time: "昨天", location: "K230+050" },
  { level: "medium", levelText: "中风险", time: "昨天", location: "K180+300" },
]);

const getAlertTagType = (level) => {
  if (level === "high") return "danger";
  if (level === "medium") return "warning";
  return "info";
};

const viewAlert = (alert) => {
  router.push("/alert-detail");
};

onMounted(() => {
  initMap();
  initCharts();
});

const initMap = () => {
  map.value = L.map("map", {
    zoomControl: false,
    attributionControl: false,
  }).setView([30.0, 104.0], 13);

  L.tileLayer(
    "http://webrd0{s}.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}",
    {
      subdomains: ["1", "2", "3", "4"],
    },
  ).addTo(map.value);
};

const initCharts = () => {
  // Pie Chart
  const pieChart = echarts.init(document.getElementById("pie-chart"));
  pieChart.setOption({
    tooltip: { trigger: "item" },
    legend: { bottom: "0", textStyle: { color: "#fff" } },
    series: [
      {
        name: "风险分布",
        type: "pie",
        radius: ["40%", "70%"],
        avoidLabelOverlap: false,
        itemStyle: { borderRadius: 10, borderColor: "#fff", borderWidth: 2 },
        label: { show: false, position: "center" },
        emphasis: { label: { show: true, fontSize: 20, fontWeight: "bold" } },
        data: [
          { value: 12, name: "高风险", itemStyle: { color: "#F56C6C" } },
          { value: 25, name: "中风险", itemStyle: { color: "#E6A23C" } },
          { value: 63, name: "低风险", itemStyle: { color: "#909399" } },
        ],
      },
    ],
  });

  // Bar Chart
  const barChart = echarts.init(document.getElementById("bar-chart"));
  barChart.setOption({
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
    grid: { left: "3%", right: "4%", bottom: "3%", containLabel: true },
    xAxis: {
      type: "value",
      splitLine: { show: false },
      axisLabel: { color: "#fff" },
    },
    yAxis: {
      type: "category",
      data: ["区域E", "区域D", "区域C", "区域B", "区域A"],
      axisLabel: { color: "#fff" },
    },
    series: [
      {
        name: "滑坡面积",
        type: "bar",
        data: [1200, 2300, 3100, 4500, 6800],
        itemStyle: { color: "#409EFF" },
      },
    ],
  });

  window.addEventListener("resize", () => {
    pieChart.resize();
    barChart.resize();
  });
};

const logout = () => {
  router.push("/login");
};

const handleFileChange = (file) => {
  selectedFile.value = file.raw;
  taskStatus.value = null;
  landslideCount.value = null;
};

const statusType = computed(() => {
  if (taskStatus.value === "completed") return "success";
  if (taskStatus.value === "failed") return "danger";
  if (taskStatus.value === "processing") return "warning";
  return "info";
});

const getStatusText = (status) => {
  const statusMap = {
    uploading: "上传中...",
    pending: "排队中...",
    processing: "分析中...",
    completed: "完成",
    failed: "失败",
    error: "错误",
  };
  return statusMap[status] || status;
};

const startAnalysis = async () => {
  if (!selectedFile.value) return;
  loading.value = true;
  taskStatus.value = "uploading";
  try {
    const uploadRes = await api.uploadImage(selectedFile.value);
    const imageId = uploadRes.data.image_id;
    const predictRes = await api.predict(imageId);
    const taskId = predictRes.data.task_id;
    taskStatus.value = "processing";
    pollInterval = setInterval(async () => {
      const statusRes = await api.getTaskStatus(taskId);
      const status = statusRes.data.status;
      taskStatus.value = status;
      if (status === "completed" || status === "failed") {
        clearInterval(pollInterval);
        loading.value = false;
        if (status === "completed") {
          landslideCount.value = statusRes.data.landslides_count;
          loadResults(taskId);
          ElMessage.success("智能分析完成！");
        } else {
          ElMessage.error("分析任务失败");
        }
      }
    }, 2000);
  } catch (error) {
    console.error(error);
    loading.value = false;
    taskStatus.value = "error";
    ElMessage.error("系统发生错误");
  }
};

const loadResults = async (taskId) => {
  try {
    const res = await api.getResults(taskId);
    const geojson = res.data;
    if (geojson.features.length > 0) {
      const layer = L.geoJSON(geojson, {
        style: {
          color: "#ff0000",
          weight: 2,
          opacity: 1,
          fillOpacity: 0.5,
        },
        onEachFeature: (feature, layer) => {
          layer.bindPopup(`置信度: ${feature.properties.confidence}`);
        },
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
  z-index: 2000;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
}

.logo {
  font-size: 20px;
  font-weight: bold;
  letter-spacing: 1px;
}

.header-center {
  flex: 1;
  display: flex;
  justify-content: center;
}

.user-info span {
  margin-right: 15px;
}

.main-content {
  flex: 1;
  position: relative;
  overflow: hidden;
  background: #000;
}

.map-container {
  width: 100%;
  height: 100%;
  z-index: 1;
}

.panel {
  position: absolute;
  top: 20px;
  bottom: 20px;
  width: 300px;
  z-index: 1000;
  display: flex;
  flex-direction: column;
  gap: 20px;
  pointer-events: none; /* Let clicks pass through gaps */
}

.panel-card {
  pointer-events: auto; /* Re-enable clicks on cards */
  background: rgba(0, 21, 41, 0.8) !important;
  border: 1px solid rgba(255, 255, 255, 0.1) !important;
  color: #fff !important;
}

.left-panel {
  left: 20px;
}

.right-panel {
  right: 20px;
}

.card-header {
  color: #409eff;
  font-weight: bold;
  font-size: 16px;
}

:deep(.el-card__header) {
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  padding: 10px 15px;
}

:deep(.el-card__body) {
  padding: 15px;
}

.stat-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
}

.stat-item {
  text-align: center;
}

.stat-value {
  font-size: 24px;
  font-weight: bold;
  color: #e6a23c;
}

.stat-label {
  font-size: 12px;
  color: #ccc;
}

.chart-container {
  height: 200px;
  width: 100%;
}

.alert-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
  max-height: 200px;
  overflow-y: auto;
}

.alert-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 13px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
  padding-bottom: 5px;
  cursor: pointer;
}

.alert-item:hover {
  background-color: rgba(255, 255, 255, 0.05);
}

.alert-time {
  color: #ccc;
}

.alert-loc {
  color: #fff;
}

.control-panel {
  position: absolute;
  bottom: 30px;
  left: 50%;
  transform: translateX(-50%);
  width: 300px;
  z-index: 1000;
}

.box-card {
  background: rgba(255, 255, 255, 0.95);
}

.file-info {
  margin-top: 10px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.file-name {
  font-size: 12px;
  color: #666;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 150px;
}
</style>
