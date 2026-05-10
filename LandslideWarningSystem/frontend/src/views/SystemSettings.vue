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
          default-active="/system-settings"
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
      <div class="settings-container">
        <el-card class="box-card">
          <template #header>
            <div class="card-header">
              <span>系统配置中心</span>
            </div>
          </template>

          <el-tabs v-model="activeTab" class="settings-tabs">
            <!-- Tab 1: Basic Settings -->
            <el-tab-pane label="基础设置" name="basic">
              <el-form
                :model="basicForm"
                label-width="120px"
                class="settings-form"
              >
                <el-form-item label="系统名称">
                  <el-input v-model="basicForm.systemName" />
                </el-form-item>
                <el-form-item label="地图默认中心">
                  <el-col :span="11">
                    <el-input
                      v-model="basicForm.mapCenterLat"
                      placeholder="纬度 (Lat)"
                    />
                  </el-col>
                  <el-col :span="2" class="text-center">-</el-col>
                  <el-col :span="11">
                    <el-input
                      v-model="basicForm.mapCenterLng"
                      placeholder="经度 (Lng)"
                    />
                  </el-col>
                </el-form-item>
                <el-form-item label="风险阈值设定">
                  <div class="slider-block">
                    <span class="demonstration"
                      >中风险 / 高风险 分界线 (置信度)</span
                    >
                    <el-slider
                      v-model="basicForm.riskThreshold"
                      :step="0.05"
                      :min="0"
                      :max="1"
                      show-input
                    />
                  </div>
                </el-form-item>
                <el-form-item label="数据自动备份">
                  <el-switch v-model="basicForm.autoBackup" />
                </el-form-item>
                <el-form-item>
                  <el-button type="primary" @click="saveBasicSettings"
                    >保存更改</el-button
                  >
                </el-form-item>
              </el-form>
            </el-tab-pane>

            <!-- Tab 2: User Management -->
            <el-tab-pane label="用户管理" name="users">
              <div class="user-toolbar">
                <el-button
                  type="primary"
                  :icon="Plus"
                  @click="dialogVisible = true"
                  >新增用户</el-button
                >
                <el-input
                  v-model="userSearch"
                  placeholder="搜索用户..."
                  class="user-search"
                  :prefix-icon="Search"
                />
              </div>
              <el-table :data="userList" stripe style="width: 100%">
                <el-table-column prop="id" label="ID" width="80" />
                <el-table-column prop="username" label="用户名" width="150" />
                <el-table-column prop="role" label="角色" width="120">
                  <template #default="scope">
                    <el-tag
                      :type="scope.row.role === 'admin' ? 'danger' : 'success'"
                    >
                      {{ scope.row.role === "admin" ? "管理员" : "操作员" }}
                    </el-tag>
                  </template>
                </el-table-column>
                <el-table-column prop="status" label="状态" width="100">
                  <template #default="scope">
                    <el-switch
                      v-model="scope.row.status"
                      active-color="#13ce66"
                      inactive-color="#ff4949"
                    />
                  </template>
                </el-table-column>
                <el-table-column prop="lastLogin" label="最后登录时间" />
                <el-table-column label="操作" width="150">
                  <template #default>
                    <el-button link type="primary" size="small">编辑</el-button>
                    <el-button link type="danger" size="small">删除</el-button>
                  </template>
                </el-table-column>
              </el-table>
            </el-tab-pane>

            <!-- Tab 3: Model Configuration -->
            <el-tab-pane label="模型配置" name="model">
              <el-form
                :model="modelForm"
                label-width="140px"
                class="settings-form"
              >
                <el-form-item label="当前激活模型">
                  <el-select
                    v-model="modelForm.activeModel"
                    placeholder="请选择模型"
                  >
                    <el-option label="HSC-HENet (推荐)" value="hsc_henet" />
                    <el-option label="DSCM-Net (实验)" value="dscm_net" />
                    <el-option label="U-Net Baseline" value="unet" />
                  </el-select>
                </el-form-item>
                <el-form-item label="推理置信度">
                  <el-slider
                    v-model="modelForm.confidence"
                    :step="0.1"
                    :min="0.1"
                    :max="0.9"
                    show-stops
                  />
                  <div class="form-tip">低于此置信度的检测结果将被过滤</div>
                </el-form-item>
                <el-form-item label="GPU 加速">
                  <el-switch
                    v-model="modelForm.useGPU"
                    active-text="开启"
                    inactive-text="关闭"
                  />
                </el-form-item>
                <el-form-item label="TTA 测试时增强">
                  <el-checkbox v-model="modelForm.tta"
                    >启用 (Horizontal Flip, Vertical Flip)</el-checkbox
                  >
                </el-form-item>
                <el-form-item>
                  <el-button type="primary" @click="saveModelSettings"
                    >应用配置</el-button
                  >
                </el-form-item>
              </el-form>
            </el-tab-pane>
          </el-tabs>
        </el-card>
      </div>
    </div>

    <!-- Add User Dialog -->
    <el-dialog v-model="dialogVisible" title="新增用户" width="30%">
      <el-form :model="newUser" label-width="80px">
        <el-form-item label="用户名">
          <el-input v-model="newUser.username" />
        </el-form-item>
        <el-form-item label="角色">
          <el-select v-model="newUser.role" placeholder="请选择">
            <el-option label="管理员" value="admin" />
            <el-option label="操作员" value="operator" />
          </el-select>
        </el-form-item>
      </el-form>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="dialogVisible = false">取消</el-button>
          <el-button type="primary" @click="addUser">确认</el-button>
        </span>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, reactive } from "vue";
import { useRouter } from "vue-router";
import { ElMessage } from "element-plus";
import { Plus, Search } from "@element-plus/icons-vue";

const router = useRouter();
const activeTab = ref("basic");
const dialogVisible = ref(false);
const userSearch = ref("");

const logout = () => {
  router.push("/login");
};

// --- Basic Settings ---
const basicForm = reactive({
  systemName: "油气管道滑坡智能预警系统",
  mapCenterLat: "30.0",
  mapCenterLng: "104.0",
  riskThreshold: 0.75,
  autoBackup: true,
});

const saveBasicSettings = () => {
  ElMessage.success("基础设置已保存");
};

// --- User Management ---
const userList = ref([
  {
    id: 1,
    username: "admin",
    role: "admin",
    status: true,
    lastLogin: "2025-12-01 10:00:00",
  },
  {
    id: 2,
    username: "operator1",
    role: "operator",
    status: true,
    lastLogin: "2025-12-01 16:30:00",
  },
  {
    id: 3,
    username: "viewer",
    role: "operator",
    status: false,
    lastLogin: "2025-12-01 09:15:00",
  },
]);

const newUser = reactive({
  username: "",
  role: "operator",
});

const addUser = () => {
  if (!newUser.username) {
    ElMessage.warning("请输入用户名");
    return;
  }
  userList.value.push({
    id: userList.value.length + 1,
    username: newUser.username,
    role: newUser.role,
    status: true,
    lastLogin: "-",
  });
  dialogVisible.value = false;
  newUser.username = "";
  ElMessage.success("用户添加成功");
};

// --- Model Settings ---
const modelForm = reactive({
  activeModel: "hsc_henet",
  confidence: 0.5,
  useGPU: true,
  tta: false,
});

const saveModelSettings = () => {
  ElMessage.success("模型配置已更新，下次分析时生效");
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

.settings-container {
  max-width: 1000px;
  margin: 0 auto;
}

.box-card {
  min-height: 500px;
}

.settings-tabs {
  margin-top: 10px;
}

.settings-form {
  max-width: 600px;
  margin-top: 20px;
}

.text-center {
  text-align: center;
}

.form-tip {
  font-size: 12px;
  color: #999;
  line-height: 1.5;
  margin-top: 5px;
}

.user-toolbar {
  display: flex;
  justify-content: space-between;
  margin-bottom: 20px;
  margin-top: 10px;
}

.user-search {
  width: 200px;
}

.slider-block {
  display: flex;
  flex-direction: column;
  width: 100%;
}

.demonstration {
  font-size: 12px;
  color: #909399;
  margin-bottom: 5px;
}
</style>
