import streamlit as st
import pandas as pd
import joblib
import os

# --- 1. จัดการ Path ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_artifacts", "gaming_model_pipeline.pkl")

st.set_page_config(page_title="Gaming Addiction Predictor", layout="wide")

@st.cache_resource
def load_assets():
    if not os.path.exists(MODEL_PATH):
        st.error(f"หาไฟล์ไม่เจอ: {MODEL_PATH}")
        st.stop()
    return joblib.load(MODEL_PATH)

try:
    pipeline = load_assets()
    # รายชื่อคอลัมน์ทั้งหมดที่โมเดล "ต้องการ" (134 คอลัมน์)
    model_features = pipeline.feature_names_in_
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# --- 2. หน้าจอ UI ---
st.title("🎮 แบบประเมินระดับการเสพติดเกม")

with st.form("main_form"):
    st.info("กรุณากรอกข้อมูลเฉพาะส่วนที่เป็นตัวเลข ระบบจะจัดการส่วนที่เหลือให้เอง")
    
    # เราจะแสดงช่องกรอก "เฉพาะตัวเลข" ที่มีอยู่ในโมเดล เพื่อไม่ให้หน้าเว็บรก
    # ส่วนที่เป็น One-Hot (0, 1) เราจะให้ค่าเริ่มต้นเป็น 0 อัตโนมัติ
    numeric_cols = [f for f in model_features if pipeline.named_steps['preprocessor'].transformers_[0][2].__contains__(f) 
                    if isinstance(pipeline.named_steps['preprocessor'].transformers_[0][2], list)]
    
    # ถ้าดึงแบบละเอียดไม่ได้ ให้โชว์คอลัมน์พื้นฐานที่สำคัญ
    if not numeric_cols:
        numeric_cols = ['Age', 'Hours', 'GAD_T', 'SWL_T', 'SPIN_T'] # หรือชื่อตาม Dataset คุณ

    cols = st.columns(4)
    user_data = {}
    
    # วนลูปสร้างช่องกรอกเฉพาะคอลัมน์ที่จำเป็น
    for i, name in enumerate(numeric_cols):
        with cols[i % 4]:
            user_data[name] = st.number_input(f"{name}", value=0.0)

    submit = st.form_submit_button("วิเคราะห์ผล")

# --- 3. การประมวลผล (หัวใจของการแก้ Error 134 features) ---
if submit:
    # 1. สร้าง DataFrame เริ่มต้นจากที่กรอก
    input_df = pd.DataFrame([user_data])
    
    # 2. 🔥 สร้างคอลัมน์ที่เหลือให้ครบ 134 คอลัมน์ และเติมเป็น 0
    for col in model_features:
        if col not in input_df.columns:
            input_df[col] = 0.0
            
    # 3. เรียงลำดับให้ตรงเป๊ะ 
    input_df = input_df[model_features]
    
    try:
        prediction = pipeline.predict(input_df)[0]
        st.success(f"### 📊 ผลการทำนายระดับการเสพติด: {prediction:.2f}")
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการทำนาย: {e}")