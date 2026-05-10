<template>
  <div class="page-container">
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
      <div class="detail-container">
        <!-- Top: Info Banner -->
        <el-alert
          :title="alertTitle"
          :type="alertType"
          :description="alertDesc"
          show-icon
          :closable="false"
          class="info-banner"
        />

        <div class="content-split">
          <!-- Left: Map -->
          <div class="left-section">
            <el-card class="box-card map-card">
              <template #header>
                <div class="card-header">
                  <span>灾害位置示意图</span>
                </div>
              </template>
              <div id="detail-map" class="map-view"></div>
            </el-card>
          </div>

          <!-- Right: Attributes & Handling -->
          <div class="right-section">
            <el-card class="box-card attr-card">
              <template #header>
                <div class="card-header">
                  <span>滑坡属性详情</span>
                </div>
              </template>
              <el-descriptions :column="1" border>
                <el-descriptions-item label="告警 ID"
                  >AL-20251201-001</el-descriptions-item
                >
                <el-descriptions-item label="风险等级">
                  <el-tag type="danger">高风险</el-tag>
                </el-descriptions-item>
                <el-descriptions-item label="检测时间"
                  >2025-12-01 10:23:45</el-descriptions-item
                >
                <el-descriptions-item label="滑坡面积"
                  >2,450 m²</el-descriptions-item
                >
                <el-descriptions-item label="中心坐标"
                  >E 103.45°, N 30.21°</el-descriptions-item
                >
                <el-descriptions-item label="影响管道段"
                  >K120+500 - K120+600</el-descriptions-item
                >
                <el-descriptions-item label="距管道最小距离"
                  >12.5 m</el-descriptions-item
                >
              </el-descriptions>
            </el-card>
          </div>
        </div>

        <!-- Bottom: Handling -->
        <div class="content-bottom">
          <el-card class="box-card handle-card">
            <template #header>
              <div class="card-header">
                <span>处置记录</span>
              </div>
            </template>
            <div class="handle-container">
              <div class="timeline-area">
                <el-timeline>
                  <el-timeline-item
                    timestamp="2025-12-01 10:23"
                    placement="top"
                    type="danger"
                  >
                    系统自动告警
                  </el-timeline-item>
                  <el-timeline-item
                    timestamp="2025-12-01 10:25"
                    placement="top"
                    type="warning"
                  >
                    管理员 [admin] 确认告警
                  </el-timeline-item>
                </el-timeline>
              </div>

              <div class="handle-form">
                <el-input
                  v-model="handleNote"
                  :rows="3"
                  type="textarea"
                  placeholder="请输入处置意见..."
                />
                <el-button
                  type="primary"
                  style="margin-top: 10px; width: 100%"
                  @click="submitHandle"
                >
                  提交处置结果
                </el-button>
              </div>
            </div>
          </el-card>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from "vue";
import { useRouter } from "vue-router";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { ElMessage } from "element-plus";

const router = useRouter();
const handleNote = ref("");

const alertTitle = computed(() => "高风险滑坡告警 - K120+500");
const alertType = computed(() => "error"); // success/info/warning/error
const alertDesc = computed(
  () => "检测到距离管道 12.5m 处存在大型滑坡风险，请立即核查！",
);

const logout = () => {
  router.push("/login");
};

const submitHandle = () => {
  if (!handleNote.value) {
    ElMessage.warning("请输入处置意见");
    return;
  }
  ElMessage.success("处置结果已提交");
  handleNote.value = "";
};

onMounted(() => {
  initMap();
});

const initMap = () => {
  const map = L.map("detail-map", {
    zoomControl: false,
    attributionControl: false,
  }).setView([30.21, 103.45], 15);

  L.tileLayer(
    "http://webrd0{s}.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}",
    {
      subdomains: ["1", "2", "3", "4"],
    },
  ).addTo(map);

  // Add a marker/polygon
  L.circle([30.21, 103.45], {
    color: "red",
    fillColor: "#f03",
    fillOpacity: 0.5,
    radius: 100,
  }).addTo(map);

  // Add pipeline line
  L.polyline(
    [
      [30.2, 103.44],
      [30.22, 103.46],
    ],
    { color: "blue", weight: 5 },
  ).addTo(map);
};
</script>

<style scoped>
.page-container {
  height: 100vh;
  display: flex;
  flex-direction: column;
  background-color: #f0f2f5;
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
  padding: 20px;
  overflow: auto;
}

.detail-container {
  max-width: 1200px;
  margin: 0 auto;
}

.info-banner {
  margin-bottom: 20px;
}

.content-middle {
  display: flex;
  gap: 20px;
  height: 400px;
  margin-bottom: 20px;
}

.content-bottom {
  height: 250px;
}

.left-section {
  flex: 2;
}

.right-section {
  flex: 1;
}

.box-card {
  height: 100%;
}

.map-card :deep(.el-card__body) {
  height: calc(100% - 60px);
  padding: 0;
}

.map-view {
  width: 100%;
  height: 100%;
}

.handle-container {
  display: flex;
  gap: 20px;
}

.timeline-area {
  flex: 1;
  border-right: 1px solid #eee;
  padding-right: 20px;
}

.handle-form {
  flex: 1;
}
</style>
