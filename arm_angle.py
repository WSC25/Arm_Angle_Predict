import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ===============================
# 頁面配置
# ===============================
st.set_page_config(page_title="中信兄弟 Arm Angle 預測系統", page_icon="⚾")

st.title("⚾ Arm Angle 手臂放球角度預測")
st.markdown("""
根據投球進階數據預測投手的 **Arm Angle**。
請在左側輸入數據(需輸入英制單位)，點擊下方按鈕進行預測。
""")

# ===============================
# 載入模型 (使用 cache 避免重複讀取)
# ===============================
@st.cache_resource
def load_model():
    # 確保你的 pkl 檔案與此 app.py 在同一目錄
    return joblib.load('arm_angle_model.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"找不到模型檔案 'arm_angle_model.pkl'，請確認檔案路徑是否正確。")
    st.stop()

# ===============================
# 側邊欄輸入欄位
# ===============================
st.sidebar.header("📥 輸入投球數據")

# 依照你原有的 predictors 順序建立輸入框
spin_axis = st.sidebar.number_input("Spin Axis", min_value=0.0, max_value=360.0, value=200.0, step=1.0)
rel_side = st.sidebar.number_input("RelSide (ft)", value=2.0, step=0.1)
rel_height = st.sidebar.number_input("RelHeight (ft)", value=5.8, step=0.1)
horz_break = st.sidebar.number_input("HorzBreak (in)", value=10.0, step=0.1)
vert_break = st.sidebar.number_input("VertBreak (in)", value=-20.0, step=0.1)
induced_vert_break = st.sidebar.number_input("InducedVertBreak (in)", value=15.0, step=0.1)

# ===============================
# 主畫面預測邏輯
# ===============================
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("🚀 開始預測", use_container_width=True):
        # 準備特徵陣列 (須符合 predictors 的順序)
        # ['spin_axis', 'RelSide', 'RelHeight','HorzBreak','VertBreak', 'InducedVertBreak']
        features = np.array([[spin_axis, rel_side, rel_height, horz_break, vert_break, induced_vert_break]])
        
        # 進行預測
        prediction = model.predict(features)[0]
        
        # 顯示結果
        st.balloons()
        st.success(f"### 預測 Arm Angle: `{prediction:.2f}°`")
        
        # 簡單的角度分類描述
        #if prediction > 70:
           # st.info("💡 投球姿勢分類預估：**高壓 (Overhand)**")
        #elif prediction > 45:
          #  st.info("💡 投球姿勢分類預估：**四分之三 (Three-Quarters)**")
        #elif prediction > 10:
         #   st.info("💡 投球姿勢分類預估：**側投 (Sidearm)**")
        #else:
         #   st.info("💡 投球姿勢分類預估：**下勾 (Submarine)**")

with col2:
    # 這裡可以放一張輔助圖示或說明
    st.markdown("### 數據參考基準")
    ref_data = {
        "指標": ["RelHeight", "RelSide", "SpinAxis"],
        "一般範圍": ["5.5 - 6.5 ft", "1.5 - 2.5 ft", "150° - 240°"]
    }
    st.table(pd.DataFrame(ref_data))

# ===============================
# 底部說明
# ===============================
st.divider()
st.caption("註：本系統僅供內部訓練與研發使用。數據準確度取決於輸入品質與模型訓練樣本。")