import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ===============================
# 頁面配置
# ===============================
st.set_page_config(page_title="Arm Angle 預測", page_icon="⚾")

st.title("Arm Angle 預測")
st.markdown("""
根據 Trackman 數據預測投手的 **Arm Angle**。
可自由切換 **英制** 或 **公制** 輸入，系統會自動處理轉換。
""")

# ===============================
# 載入模型
# ===============================
@st.cache_resource
def load_model():
    return joblib.load('arm_angle_model.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ 無法載入模型: {e}")
    st.stop()

# ===============================
# 側邊欄：單位切換與輸入
# ===============================
st.sidebar.header("📥 輸入投球數據")

# --- 單位切換選擇器 ---
unit_system = st.sidebar.radio("選擇輸入單位系統", ["英制 (ft/in)", "公制 (m/cm)"])

if unit_system == "英制 (ft/in)":
    v1 = st.sidebar.number_input("Spin Axis (旋轉軸 0-360)", value=200.0, step=1.0)
    v2 = st.sidebar.number_input("RelSide (ft)", value=2.0, step=0.1)
    v3 = st.sidebar.number_input("RelHeight (ft)", value=5.8, step=0.1)
    v4 = st.sidebar.number_input("HorzBreak (in)", value=10.0, step=0.1)
    v5 = st.sidebar.number_input("VertBreak (in)", value=-20.0, step=0.1)
    v6 = st.sidebar.number_input("InducedVertBreak (in)", value=15.0, step=0.1)
else:
    # 公制輸入 (給予對應的公制預設值，例如 5.8ft 換算約 1.77m)
    v1 = st.sidebar.number_input("Spin Axis (旋轉軸 0-360)", value=200.0, step=1.0)
    v2_m = st.sidebar.number_input("RelSide (公尺 m)", value=0.61, step=0.01)
    v3_m = st.sidebar.number_input("RelHeight (公尺 m)", value=1.77, step=0.01)
    v4_cm = st.sidebar.number_input("HorzBreak (公分 cm)", value=25.4, step=0.1)
    v5_cm = st.sidebar.number_input("VertBreak (公分 cm)", value=-50.8, step=0.1)
    v6_cm = st.sidebar.number_input("InducedVertBreak (公分 cm)", value=38.1, step=0.1)
    
    # --- 公制轉英制公式 ---
    v2 = v2_m * 3.28084
    v3 = v3_m * 3.28084
    v4 = v4_cm * 0.3937
    v5 = v5_cm * 0.3937
    v6 = v6_cm * 0.3937

# ===============================
# 主畫面預測邏輯
# ===============================
predictors = ['spin_axis', 'RelSide', 'RelHeight', 'HorzBreak', 'VertBreak', 'InducedVertBreak']

if st.button("🚀 開始預測", use_container_width=True):
    try:
        # 建立預測用 DataFrame (此時數據皆已轉為英制)
        input_df = pd.DataFrame(
            [[v1, v2, v3, v4, v5, v6]], 
            columns=predictors
        ).astype(float)
        
        # 進行預測
        prediction = model.predict(input_df)[0]
        
        # 顯示結果
        st.success(f"### 預測 Arm Angle: `{prediction:.2f}°`")
        
        # 顯示換算後的英制參考值 (方便確認轉換是否正確)
        if unit_system == "公制 (m/cm)":
            with st.expander("查看自動換算的英制數值 (模型計算用)"):
                st.write(f"RelSide: {v2:.2f} ft | RelHeight: {v3:.2f} ft")
                st.write(f"HorzBreak: {v4:.2f} in | VertBreak: {v5:.2f} in | IVB: {v6:.2f} in")
        
    except Exception as e:
        st.error(f"預測執行失敗：{e}")

# ===============================
# 底部說明
# ===============================
st.divider()
st.caption("本模型輸出之 Arm Angle 為基於大聯盟數據之預測值。由於預測模型本質上存在統計誤差，加上現有技術難以對中職所有投球瞬間進行精確物理驗證，結果僅供內部輔助分析參考。")






