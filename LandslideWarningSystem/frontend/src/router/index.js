import { createRouter, createWebHistory } from "vue-router";

const routes = [
  { path: "/", component: () => import("../views/HomeCN.vue") },
  {
    path: "/data-management",
    component: () => import("../views/DataManagement.vue"),
  },
  {
    path: "/sample-augmentation",
    component: () => import("../views/SampleAugmentation.vue"),
  },
  {
    path: "/system-settings",
    component: () => import("../views/SystemSettings.vue"),
  },
  {
    path: "/alert-detail",
    component: () => import("../views/AlertDetail.vue"),
  },
  { path: "/en", component: () => import("../views/Home.vue") },
  { path: "/login", component: () => import("../views/Login.vue") },
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

export default router;
