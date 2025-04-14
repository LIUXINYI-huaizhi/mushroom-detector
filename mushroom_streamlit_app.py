# mushroom_streamlit_app.py
import streamlit as st
from PIL import Image
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
import os
import urllib.request

# 模型文件下载
MODEL_PATH = "mushroom_resnet50.pth"
MODEL_URL = "https://huggingface.co/LIUXINYI-huaizhi/mushroom-detector/resolve/main/mushroom_resnet50.pth"

def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("正在下载模型..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# 模型加载
def load_model(model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = {
        k.replace('model.', ''): v
        for k, v in checkpoint['model_state_dict'].items()
    }
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 1),
        nn.Sigmoid()
    )
    model.load_state_dict(state_dict)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((checkpoint['input_size'], checkpoint['input_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=checkpoint['transform_mean'], std=checkpoint['transform_std'])
    ])
    return model, transform, checkpoint['class_names']

# 推理函数
def predict_image(image, model, transform):
    image = image.convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        prob = output.item()
        pred = 1 if prob > 0.5 else 0
    return pred, prob

# Streamlit 页面配置
st.set_page_config(page_title="蘑菇毒性识别系统", layout="centered")
st.markdown("""
<style>
body {
    background-color: #f5f5f5;
}
.stApp {
    max-width: 800px;
    margin: auto;
    font-family: 'Arial';
}
.title-box {
    background: linear-gradient(90deg, #ff416c, #ff4b2b);
    color: white;
    padding: 1em;
    border-radius: 12px;
    text-align: center;
}
.warning-box {
    background-color: #ffcccc;
    color: red;
    border: 2px solid red;
    border-radius: 10px;
    padding: 1em;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title-box'><h2>🍄 蘑菇毒性识别系统</h2></div>", unsafe_allow_html=True)
st.markdown("上传蘑菇图片，我们将判断是否为有毒品种！")

# 示例图片
st.markdown("**示例图测试：**")
col1, col2 = st.columns(2)
with col1:
    if st.button("🚫 有毒蘑菇示例"):
        image = Image.open("toxic_example.jpg")
        st.image(image, caption="示例：有毒蘑菇", use_container_width=True)
        download_model()
        model, transform, class_names = load_model(MODEL_PATH)
        pred, prob = predict_image(image, model, transform)
        if pred == 1:
            st.markdown("<div class='warning-box'>⚠️ 预测结果：<b>有毒蘑菇</b>（置信度：{:.2f}）</div>".format(prob), unsafe_allow_html=True)
        else:
            st.success(f"✅ 可食蘑菇（置信度：{prob:.2f}）")

with col2:
    if st.button("✅ 可食蘑菇示例"):
        image = Image.open("edible_example.jpg")
        st.image(image, caption="示例：可食蘑菇", use_container_width=True)
        download_model()
        model, transform, class_names = load_model(MODEL_PATH)
        pred, prob = predict_image(image, model, transform)
        if pred == 1:
            st.markdown("<div class='warning-box'>⚠️ 预测结果：<b>有毒蘑菇</b>（置信度：{:.2f}）</div>".format(prob), unsafe_allow_html=True)
        else:
            st.success(f"✅ 可食蘑菇（置信度：{prob:.2f}）")

# 上传图像识别
st.markdown("---")
uploaded_file = st.file_uploader("或者上传你自己的蘑菇照片：", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="上传的图片", use_container_width=True)
    download_model()
    model, transform, class_names = load_model(MODEL_PATH)
    pred, prob = predict_image(image, model, transform)
    if pred == 1:
        st.markdown("<div class='warning-box'>⚠️ 预测结果：<b>有毒蘑菇</b>（置信度：{:.2f}）</div>".format(prob), unsafe_allow_html=True)
    else:
        st.success(f"✅ 可食蘑菇（置信度：{prob:.2f}）")
