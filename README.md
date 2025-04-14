# 🍄 蘑菇毒性识别系统（Mushroom Toxicity Classifier）

> 使用 PyTorch + ResNet50 + Streamlit 构建的图像分类网页应用，可上传蘑菇图片自动判断是否为毒蘑菇。

![Streamlit Status](https://img.shields.io/badge/deployed-yes-brightgreen)

---

## 🚀 在线体验地址

👉 [点击访问 Streamlit Web App](https://liuxinyi-huaizhi-mushroom-detector.streamlit.app)

---

## 🧠 模型说明

- **主干网络**：ResNet-50（使用 ImageNet 预训练）
- **输出形式**：Sigmoid 二分类（毒 / 不毒）
- **输入图片尺寸**：224×224×3
- **推理逻辑**：模型输出 > 0.5 判为有毒

---

## 📦 安装与运行

```bash
# 安装依赖
pip install -r requirements.txt

# 启动本地服务
streamlit run mushroom_streamlit_app.py
```

---

## 📁 项目结构

```
├── mushroom_streamlit_app.py   # 主程序
├── requirements.txt            # 依赖文件
├── mushroom_resnet50.pth       # 模型文件（首次自动下载）
├── static/
│   ├── toxic_example.jpg       # 示例图：有毒
│   └── edible_example.jpg      # 示例图：可食
```

---

## 📸 页面功能预览

- [x] 上传蘑菇图片自动分类
- [x] 示例按钮：毒蘑菇 / 可食蘑菇
- [x] 红色警告提示框（高置信度毒性）
- [x] 模型文件自动下载，无需手动放置
- [x] 页面渐变标题 / 居中布局美化

---

## 📤 部署说明（Streamlit Cloud）

1. 将代码上传至公开 GitHub 仓库
2. 登录 https://streamlit.io/cloud
3. 选择你的仓库 → 设置主文件：`mushroom_streamlit_app.py`
4. 部署后自动生成可访问链接

---

## 🙌 作者

- 👤 [LIUXINYI-huaizhi](https://github.com/LIUXINYI-huaizhi)
- 💬 若你喜欢本项目，可点个 star ⭐ 支持一下！

---

## 📄 License

MIT License. 模型及页面自由使用，欢迎扩展与二次开发。
