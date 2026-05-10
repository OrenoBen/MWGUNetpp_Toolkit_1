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
          default-active="/data-management"
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
      <el-card class="content-card">
        <div class="toolbar">
          <div class="search-area">
            <el-input placeholder="请输入文件名搜索" v-model="searchQuery" style="width: 200px; margin-right: 10px;" />
            <el-date-picker
              v-model="dateRange"
              type="daterange"
              range-separator="至"
              start-placeholder="开始日期"
              end-placeholder="结束日期"
              style="margin-right: 10px;"
            />
            <el-button type="primary" :icon="Search">搜索</el-button>
          </div>
          <div class="action-area">
            <el-button type="success" :icon="Upload">上传数据</el-button>
            <el-button type="danger" :icon="Delete">批量删除</el-button>
          </div>
        </div>

        <el-table :data="tableData" stripe style="width: 100%">
          <el-table-column type="selection" width="55" />
          <el-table-column label="缩略图" width="120">
            <template #default="scope">
              <el-image style="width: 80px; height: 60px" :src="scope.row.thumbnail" fit="cover" />
            </template>
          </el-table-column>
          <el-table-column prop="filename" label="文件名" />
          <el-table-column prop="source" label="数据源" width="120" />
          <el-table-column prop="resolution" label="分辨率" width="120" />
          <el-table-column prop="date" label="采集时间" width="180" />
          <el-table-column label="操作" width="250">
            <template #default>
              <el-button size="small" :icon="View">预览</el-button>
              <el-button size="small" :icon="Download">下载</el-button>
              <el-button size="small" type="danger" :icon="Delete">删除</el-button>
            </template>
          </el-table-column>
        </el-table>

        <div class="pagination-container">
          <el-pagination background layout="prev, pager, next" :total="100" />
        </div>
      </el-card>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import { useRouter } from 'vue-router';
import { Search, Upload, Delete, View, Download } from '@element-plus/icons-vue';

const router = useRouter();
const searchQuery = ref('');
const dateRange = ref('');

const logout = () => {
  router.push('/login');
};

const tableData = ref([
  {
    thumbnail: 'https://via.placeholder.com/80x60',
    filename: 'S2A_20251201_T50RKU.tif',
    source: 'Sentinel-2',
    resolution: '10m',
    date: '2025-12-01 10:30:00'
  },
  {
    thumbnail: 'https://via.placeholder.com/80x60',
    filename: 'GF1_20251201_E103.5_N30.2.tiff',
    source: 'Gaofen-1',
    resolution: '2m',
    date: '2025-12-01 14:15:00'
  },
  {
    thumbnail: 'https://via.placeholder.com/80x60',
    filename: 'DEM_Sichuan_2025.tif',
    source: 'ALOS DEM',
    resolution: '12.5m',
    date: '2025-12-01 09:00:00'
  },
  {
    thumbnail: 'https://via.placeholder.com/80x60',
    filename: 'Pipeline_Vector_2025.shp',
    source: 'Vector',
    resolution: '-',
    date: '2025-12-01 16:45:00'
  },
  {
    thumbnail: 'https://via.placeholder.com/80x60',
    filename: 'S2B_20251201_T50RKU.tif',
    source: 'Sentinel-2',
    resolution: '10m',
    date: '2025-12-01 11:05:00'
  }
]);
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
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
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
  padding: 20px;
  overflow: auto;
}

.content-card {
  min-height: 100%;
}

.toolbar {
  display: flex;
  justify-content: space-between;
  margin-bottom: 20px;
}

.pagination-container {
  margin-top: 20px;
  display: flex;
  justify-content: flex-end;
}
</style>