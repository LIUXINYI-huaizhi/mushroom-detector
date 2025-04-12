import streamlit as st
from PIL import Image
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms

def load_model(model_path):
    checkpoint = torch.load(model_path, map_location='cpu')

    # 修复键名：移除 "model." 前缀
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

def predict_image(image, model, transform):
    image = image.convert('RGB')
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        prob = output.item()
        pred = 1 if prob > 0.5 else 0

    return pred, prob

# Streamlit 页面
st.set_page_config(page_title="蘑菇毒性识别系统", layout="centered")
st.title("🍄 蘑菇毒性识别系统")

st.markdown("上传一张蘑菇图片，系统将预测它是否为毒蘑菇。")
uploaded_file = st.file_uploader("请选择一张图片：", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="上传的图片", use_column_width=True)

    with st.spinner("正在识别中，请稍候..."):
        model, transform, class_names = load_model("mushroom_resnet50.pth")
        pred, prob = predict_image(image, model, transform)
        result_text = class_names[pred]
        confidence = round(prob, 4)

    st.success(f"预测结果：**{result_text}**（置信度：{confidence}）")
