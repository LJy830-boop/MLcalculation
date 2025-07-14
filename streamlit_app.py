# coding: utf-8
"""
电池寿命预测系统 - Streamlit应用（优化版）
该脚本是为Streamlit Cloud部署优化的主应用文件
"""

import os
import sys
import warnings

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 忽略警告
warnings.filterwarnings('ignore')

# 设置环境变量
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 减少TensorFlow日志

import streamlit as st

# 页面配置
st.set_page_config(
    page_title="电池寿命预测系统",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/battery-prediction',
        'Report a bug': 'https://github.com/your-repo/battery-prediction/issues',
        'About': """
        # 电池寿命预测系统
        
        基于机器学习的电池健康状态(SOH)和剩余使用寿命(RUL)预测系统。
        
        **功能特点:**
        - 数据预处理和探索性分析
        - 多种机器学习算法支持
        - 远程服务器训练
        - 实时预测和可视化
        
        **开发团队:** 浙江锋锂新能源科技有限公司-唐光盛团队
        """
    }
)

# 检查依赖
@st.cache_data
def check_dependencies():
    """检查必要的依赖是否安装"""
    missing_deps = []
    
    try:
        import pandas
    except ImportError:
        missing_deps.append("pandas")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import sklearn
    except ImportError:
        missing_deps.append("scikit-learn")
    
    try:
        import matplotlib
    except ImportError:
        missing_deps.append("matplotlib")
    
    return missing_deps

# 检查依赖
missing_deps = check_dependencies()
if missing_deps:
    st.error(f"缺少必要的依赖包: {', '.join(missing_deps)}")
    st.info("请安装缺少的依赖包后重新运行应用")
    st.stop()

# 导入主应用
try:
    # 尝试导入带服务器功能的应用
    from battery_prediction_app_with_server import *
    
    # 添加部署信息
    st.sidebar.markdown("---")
    st.sidebar.info("🌐 已部署到Streamlit Cloud")
    
    # 添加使用提示
    if 'show_help' not in st.session_state:
        st.session_state.show_help = True
    
    if st.session_state.show_help:
        with st.expander("📖 使用指南", expanded=True):
            st.markdown("""
            ### 快速开始
            
            1. **数据上传**: 在左侧选择"1. 数据上传"，上传您的电池数据文件（CSV或Excel格式）
            2. **数据预处理**: 配置数据列映射和预处理选项
            3. **特征提取**: 提取电池特征用于模型训练
            4. **模型训练**: 选择算法和参数训练预测模型
            5. **预测分析**: 查看SOH预测和RUL计算结果
            6. **服务器连接**: （可选）连接远程服务器进行大规模训练
            
            ### 数据格式要求
            
            您的数据文件应包含以下列：
            - **循环次数**: 电池充放电循环编号
            - **电压**: 电池电压值
            - **电流**: 电池电流值  
            - **时间**: 时间戳或序列
            - **容量**: 电池容量（可选）
            - **温度**: 电池温度（可选）
            
            ### 注意事项
            
            - 文件大小限制: 100MB
            - 支持格式: CSV, Excel (.xlsx, .xls)
            - 建议数据量: 1000+ 行以获得更好的预测效果
            """)
            
            if st.button("关闭使用指南"):
                st.session_state.show_help = False
                st.rerun()

except ImportError as e:
    st.error(f"导入应用模块失败: {str(e)}")
    st.info("请确保所有必要的文件都已正确上传")
    
    # 显示基本信息
    st.title("🔋 电池寿命预测系统")
    st.markdown("""
    ## 系统暂时不可用
    
    应用正在初始化中，请稍后刷新页面重试。
    
    如果问题持续存在，请联系技术支持。
    
    ### 联系信息
    - 开发团队: 浙江锋锂新能源科技有限公司-唐光盛团队
    - 技术支持: [your-email@example.com]
    """)

except Exception as e:
    st.error(f"应用启动失败: {str(e)}")
    st.info("请刷新页面重试，或联系技术支持")
    
    # 显示错误详情（仅在开发模式）
    if st.secrets.get("app", {}).get("debug_mode", False):
        st.exception(e)

