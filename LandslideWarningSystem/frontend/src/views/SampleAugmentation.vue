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
          default-active="/sample-augmentation"
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
      <div class="split-container">
        <!-- Left: Config -->
        <div class="panel left-panel">
          <el-card class="box-card">
            <template #header>
              <div class="card-header">
                <span>WUT-Seg 增强配置</span>
              </div>
            </template>
            <el-form label-position="top">
              <el-form-item label="选择数据集">
                <el-select v-model="dataset" placeholder="请选择">
                  <el-option label="Landslide4Sense-Train" value="l4s_train" />
                  <el-option label="Landslide4Sense-Val" value="l4s_val" />
                </el-select>
              </el-form-item>
              <el-form-item label="生成样本数量">
                <el-input-number
                  v-model="sampleCount"
                  :min="100"
                  :max="10000"
                  :step="100"
                />
              </el-form-item>
              <el-form-item label="增强策略">
                <el-checkbox-group v-model="strategies">
                  <el-checkbox label="旋转/翻转" />
                  <el-checkbox label="噪声注入" />
                  <el-checkbox label="WGAN-GP 生成" />
                </el-checkbox-group>
              </el-form-item>
              <el-form-item>
                <el-button
                  type="primary"
                  style="width: 100%"
                  @click="startAugmentation"
                  :loading="loading"
                >
                  开始生成
                </el-button>
              </el-form-item>
            </el-form>

            <div class="progress-area" v-if="loading || completed">
              <p>{{ progressText }}</p>
              <el-progress :percentage="percentage" :status="progressStatus" />
            </div>
          </el-card>
        </div>

        <!-- Right: Result -->
        <div class="panel right-panel">
          <el-card class="box-card full-height">
            <template #header>
              <div class="card-header">
                <span>增强效果预览</span>
              </div>
            </template>

            <div class="charts-area">
              <div class="chart-box">
                <h4>原始样本分布</h4>
                <div id="hist-before" class="chart"></div>
              </div>
              <div class="chart-box">
                <h4>增强后样本分布</h4>
                <div id="hist-after" class="chart"></div>
              </div>
            </div>

            <div class="sample-grid">
              <h4>生成样本预览</h4>
              <div class="grid-container">
                <div
                  class="grid-item"
                  v-for="(sample, index) in sampleImages"
                  :key="index"
                >
                  <el-image :src="sample.url" fit="cover" class="sample-img">
                    <template #placeholder>
                      <div class="image-slot">加载中...</div>
                    </template>
                  </el-image>
                  <div class="sample-label">{{ sample.label }}</div>
                </div>
              </div>
            </div>
          </el-card>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, nextTick } from "vue";
import { useRouter } from "vue-router";
import * as echarts from "echarts";
import { ElMessage } from "element-plus";

const router = useRouter();
const dataset = ref("l4s_train");
const sampleCount = ref(1000);
const strategies = ref(["WGAN-GP 生成"]);
const loading = ref(false);
const completed = ref(false);
const percentage = ref(0);
const progressText = ref("准备中...");
const progressStatus = ref("");
const sampleImages = ref([
  {
    url: "https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=200&h=200",
    label: "待处理样本 01",
  },
  {
    url: "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?auto=format&fit=crop&w=200&h=200",
    label: "待处理样本 02",
  },
  {
    url: "https://images.unsplash.com/photo-1470770841072-f978cf4d019e?auto=format&fit=crop&w=200&h=200",
    label: "待处理样本 03",
  },
  {
    url: "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?auto=format&fit=crop&w=200&h=200",
    label: "待处理样本 04",
  },
]);

const logout = () => {
  router.push("/login");
};

const startAugmentation = () => {
  loading.value = true;
  completed.value = false;
  percentage.value = 0;
  progressStatus.value = "";
  progressText.value = "正在初始化 WUT-Seg 模型...";
  sampleImages.value = [];

  // Mock process
  let p = 0;
  const timer = setInterval(() => {
    p += 10;
    percentage.value = p;
    if (p < 30) progressText.value = "正在加载数据...";
    else if (p < 80) progressText.value = "正在生成对抗样本...";
    else progressText.value = "正在保存结果...";

    if (p >= 100) {
      clearInterval(timer);
      loading.value = false;
      completed.value = true;
      progressStatus.value = "success";
      progressText.value = "增强任务完成！";
      ElMessage.success("样本增强完成，新增 1000 个样本");

      // Populate mock images
      sampleImages.value = [
        {
          url: "https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=200&h=200",
          label: "原始样本 01",
        },
        {
          url: "https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=200&h=200&flip=h",
          label: "水平翻转 (Aug)",
        },
        {
          url: "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?auto=format&fit=crop&w=200&h=200",
          label: "原始样本 02",
        },
        {
          url: "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?auto=format&fit=crop&w=200&h=200&bri=-20",
          label: "亮度调整 (Aug)",
        },
        {
          url: "https://images.unsplash.com/photo-1470770841072-f978cf4d019e?auto=format&fit=crop&w=200&h=200",
          label: "WGAN 生成 01",
        },
        {
          url: "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?auto=format&fit=crop&w=200&h=200",
          label: "WGAN 生成 02",
        },
        {
          url: "https://images.unsplash.com/photo-1501785888041-af3ef285b470?auto=format&fit=crop&w=200&h=200",
          label: "噪声注入 01",
        },
        {
          url: "https://images.unsplash.com/photo-1472214103451-9374bd1c798e?auto=format&fit=crop&w=200&h=200",
          label: "旋转 90° (Aug)",
        },
      ];

      initCharts(true);
    }
  }, 500);
};

onMounted(() => {
  initCharts(false);
});

const initCharts = (showAfter) => {
  const chartBefore = echarts.init(document.getElementById("hist-before"));
  chartBefore.setOption({
    xAxis: { type: "category", data: ["滑坡", "非滑坡"] },
    yAxis: { type: "value" },
    series: [{ data: [300, 2000], type: "bar", color: "#909399" }],
  });

  const chartAfter = echarts.init(document.getElementById("hist-after"));
  chartAfter.setOption({
    xAxis: { type: "category", data: ["滑坡", "非滑坡"] },
    yAxis: { type: "value" },
    series: [
      {
        data: showAfter ? [1300, 2000] : [0, 0],
        type: "bar",
        color: "#67C23A",
      },
    ],
  });
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
  overflow: hidden;
}

.split-container {
  display: flex;
  gap: 20px;
  height: 100%;
}

.left-panel {
  width: 350px;
  flex-shrink: 0;
}

.right-panel {
  flex: 1;
}

.full-height {
  height: 100%;
  display: flex;
  flex-direction: column;
}

:deep(.el-card__body) {
  flex: 1;
  display: flex;
  flex-direction: column;
}

.charts-area {
  display: flex;
  gap: 20px;
  height: 250px;
  margin-bottom: 20px;
}

.chart-box {
  flex: 1;
  display: flex;
  flex-direction: column;
}

.chart {
  flex: 1;
}

.sample-grid {
  flex: 1;
}

.grid-container {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 15px;
  margin-top: 10px;
}

.grid-item {
  aspect-ratio: 1;
  background-color: #f5f7fa;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  border-radius: 8px;
  overflow: hidden;
  border: 1px solid #e4e7ed;
  transition: all 0.3s;
}

.grid-item:hover {
  transform: translateY(-5px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.sample-img {
  width: 100%;
  height: calc(100% - 30px);
}

.sample-label {
  height: 30px;
  line-height: 30px;
  font-size: 12px;
  color: #606266;
  background: #fff;
  width: 100%;
  text-align: center;
  border-top: 1px solid #e4e7ed;
}

.image-slot {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 100%;
  height: 100%;
  background: #f5f7fa;
  color: #909399;
  font-size: 12px;
}

.placeholder-img {
  color: #999;
  font-size: 12px;
}

.progress-area {
  margin-top: 30px;
}
</style>
