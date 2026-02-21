import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ===============================
# 頁面配置
# ===============================
st.set_page_config(page_title="Arm Angle 預測", page_icon="⚾")

st.title("Arm Angle 手臂放球角度預測")
st.markdown("""
根據Trackman數據預測投手的 **Arm Angle**。
請在左側輸入數據(需輸入英制單位)，點擊下方按鈕進行預測。
""")

# ===============================
# 載入模型
# ===============================
@st.cache_resource
def load_model():
    # 確保 arm_angle_model.pkl 放在 GitHub 根目錄或與此檔同資料夾
    return joblib.load('arm_angle_model.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ 無法載入模型: {e}")
    st.info("請檢查 'arm_angle_model.pkl' 是否已上傳至 GitHub 且檔名正確。")
    st.stop()

# ===============================
# 側邊欄輸入欄位
# ===============================
st.sidebar.header("📥 輸入投球數據")

# 這裡建立輸入變數，對應你的 predictors
v1 = st.sidebar.number_input("Spin Axis (旋轉軸)", min_value=0.0, max_value=360.0, value=200.0, step=1.0)
v2 = st.sidebar.number_input("RelSide (放球側向距離 ft)", value=2.0, step=0.1)
v3 = st.sidebar.number_input("RelHeight (放球高度 ft)", value=5.8, step=0.1)
v4 = st.sidebar.number_input("HorzBreak (水平位移 in)", value=10.0, step=0.1)
v5 = st.sidebar.number_input("VertBreak (垂直位移 in)", value=-20.0, step=0.1)
v6 = st.sidebar.number_input("InducedVertBreak (誘發垂直位移 in)", value=15.0, step=0.1)

# ===============================
# 主畫面預測邏輯
# ===============================
# 定義與訓練時完全一致的欄位順序
predictors = ['spin_axis', 'RelSide', 'RelHeight', 'HorzBreak', 'VertBreak', 'InducedVertBreak']

if st.button("🚀 開始預測", use_container_width=True):
    try:
        # 修正：將輸入封裝成 DataFrame 並指定欄位名稱
        input_df = pd.DataFrame(
            [[v1, v2, v3, v4, v5, v6]], 
            columns=predictors
        ).astype(float)
        
        # 進行預測
        prediction = model.predict(input_df)[0]
        
        # 顯示結果
        #st.balloons()
        st.success(f"### 預測 Arm Angle: `{prediction:.2f}°`")
        
    except Exception as e:
        st.error(f"預測執行失敗：{e}")
        st.info("建議：請確認 requirements.txt 中的 scikit-learn 版本是否與本地端一致。")

# ===============================
# 底部說明
# ===============================
st.divider()
st.caption("註：本系統僅供內部訓練與研發使用。數據準確度取決於輸入品質與模型訓練樣本。")




