<template>
  <div class="login-container">
    <div class="login-content">
      <div class="login-header">
        <h1 class="system-title">油气管道滑坡智能预警系统</h1>
        <p class="system-subtitle">基于深度学习的监测与分析平台</p>
      </div>

      <el-card class="login-card">
        <h2 class="login-title">用户登录</h2>
        <el-form
          :model="loginForm"
          :rules="rules"
          ref="loginFormRef"
          size="large"
          @keyup.enter="handleLogin"
        >
          <el-form-item prop="username">
            <el-input
              v-model="loginForm.username"
              placeholder="用户名"
              :prefix-icon="User"
            />
          </el-form-item>
          <el-form-item prop="password">
            <el-input
              v-model="loginForm.password"
              type="password"
              placeholder="密码"
              :prefix-icon="Lock"
              show-password
            />
          </el-form-item>
          <el-button
            type="primary"
            class="login-button"
            :loading="loading"
            @click="handleLogin"
          >
            立即登录
          </el-button>
        </el-form>
      </el-card>

      <div class="login-footer">
        <p>© 2026 Landslide Warning System. All Rights Reserved.</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive } from "vue";
import { useRouter } from "vue-router";
import { User, Lock } from "@element-plus/icons-vue";
import { ElMessage } from "element-plus";

const router = useRouter();
const loginFormRef = ref(null);
const loading = ref(false);

const loginForm = reactive({
  username: "admin",
  password: "",
});

const rules = {
  username: [{ required: true, message: "请输入用户名", trigger: "blur" }],
  password: [{ required: true, message: "请输入密码", trigger: "blur" }],
};

const handleLogin = async () => {
  if (!loginFormRef.value) return;

  await loginFormRef.value.validate((valid) => {
    if (valid) {
      loading.value = true;
      // 模拟登录请求
      setTimeout(() => {
        loading.value = false;
        ElMessage.success("登录成功");
        router.push("/");
      }, 800);
    }
  });
};
</script>

<style scoped>
.login-container {
  height: 100vh;
  width: 100vw;
  display: flex;
  justify-content: center;
  align-items: center;
  background: linear-gradient(135deg, #001529 0%, #003a70 100%);
  position: relative;
  overflow: hidden;
}

/* 背景装饰 */
.login-container::before {
  content: "";
  position: absolute;
  top: -100px;
  left: -100px;
  width: 300px;
  height: 300px;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 50%;
}

.login-container::after {
  content: "";
  position: absolute;
  bottom: -50px;
  right: -50px;
  width: 200px;
  height: 200px;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 50%;
}

.login-content {
  width: 420px;
  z-index: 1;
  padding: 20px;
}

.login-header {
  text-align: center;
  margin-bottom: 40px;
  color: white;
}

.system-title {
  font-size: 32px;
  font-weight: bold;
  margin: 0 0 10px 0;
  text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
  letter-spacing: 2px;
}

.system-subtitle {
  font-size: 16px;
  opacity: 0.8;
  margin: 0;
  letter-spacing: 1px;
}

.login-card {
  border: none;
  border-radius: 12px;
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
}

.login-title {
  text-align: center;
  margin-bottom: 30px;
  color: #303133;
  font-weight: 600;
  font-size: 20px;
}

.login-button {
  width: 100%;
  font-weight: bold;
  letter-spacing: 2px;
  background: linear-gradient(90deg, #1890ff 0%, #096dd9 100%);
  border: none;
  height: 45px;
  font-size: 16px;
  margin-top: 10px;
}

.login-button:hover {
  background: linear-gradient(90deg, #40a9ff 0%, #096dd9 100%);
  opacity: 0.9;
  transform: translateY(-1px);
  transition: all 0.3s;
}

.login-footer {
  margin-top: 30px;
  text-align: center;
  color: rgba(255, 255, 255, 0.5);
  font-size: 12px;
}
</style>
