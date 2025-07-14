# coding: utf-8
"""
电池寿命预测模型 - Streamlit应用（带服务器连接功能）
该脚本实现了电池寿命预测模型的Streamlit界面，允许用户上传数据、训练模型、可视化预测结果，
并支持连接远程服务器进行模型训练。
"""

import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib
import warnings
import time
import json
warnings.filterwarnings('ignore')

# 导入自定义模块
from data_preprocessing_pipeline import BatteryDataPreprocessor
from exploratory_data_analysis import BatteryDataExplorer
from feature_extraction import BatteryFeatureExtractor
from prediction_models import BatteryPredictionModel
from model_evaluation import ModelEvaluator
from server_connection import (
    ServerConnectionManager, 
    FileTransferManager, 
    RemoteCommandExecutor, 
    ServerConfigManager
)

# 配置页面
st.set_page_config(
    page_title="电池寿命预测系统",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 配置上传文件存储路径
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
MODELS_FOLDER = 'models'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

# 确保目录存在
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, MODELS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# 初始化会话状态
if 'data' not in st.session_state:
    st.session_state.data = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_name' not in st.session_state:
    st.session_state.model_name = None
if 'feature_cols' not in st.session_state:
    st.session_state.feature_cols = None
if 'target_col' not in st.session_state:
    st.session_state.target_col = None
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1

# 服务器连接相关状态
if 'server_manager' not in st.session_state:
    st.session_state.server_manager = ServerConnectionManager()
if 'file_transfer' not in st.session_state:
    st.session_state.file_transfer = FileTransferManager(st.session_state.server_manager)
if 'remote_executor' not in st.session_state:
    st.session_state.remote_executor = RemoteCommandExecutor(st.session_state.server_manager)
if 'config_manager' not in st.session_state:
    st.session_state.config_manager = ServerConfigManager()
if 'server_configs' not in st.session_state:
    st.session_state.server_configs = st.session_state.config_manager.load_config()

# 辅助函数
def allowed_file(filename):
    """检查文件类型是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_image_base64(fig):
    """将matplotlib图形转换为base64编码"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    return img_str

def display_connection_status():
    """显示连接状态"""
    if st.session_state.server_manager.is_connected:
        st.success("🟢 已连接到服务器")
        conn_info = st.session_state.server_manager.connection_info
        st.info(f"服务器: {conn_info.get('username')}@{conn_info.get('host')}:{conn_info.get('port')}")
    else:
        st.error("🔴 未连接到服务器")

# 侧边栏导航
st.sidebar.title("电池寿命预测系统")
st.sidebar.image("https://img.icons8.com/color/96/000000/battery-level.png", width=100)

step = st.sidebar.radio(
    "导航",
    ["1. 数据上传", "2. 数据预处理", "3. 探索性分析", "4. 特征提取", 
     "5. 模型训练", "6. 预测与评估", "7. 模型优化", "8. 服务器连接"],
    index=st.session_state.current_step - 1
)

st.session_state.current_step = int(step[0])

# 在侧边栏显示连接状态
st.sidebar.markdown("---")
st.sidebar.subheader("服务器状态")
display_connection_status()

# 1-7步骤保持原有逻辑（这里省略，与原代码相同）
# ... [原有的1-7步骤代码] ...

# 8. 服务器连接页面
if st.session_state.current_step == 8:
    st.title("8. 服务器连接")
    st.write("配置和管理远程服务器连接，支持远程模型训练")
    
    # 创建标签页
    tab1, tab2, tab3, tab4 = st.tabs(["服务器配置", "连接管理", "远程训练", "任务监控"])
    
    with tab1:
        st.subheader("服务器配置")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 添加新服务器")
            
            with st.form("add_server_form"):
                server_name = st.text_input("服务器名称", placeholder="例如: 训练服务器1")
                server_host = st.text_input("服务器地址", placeholder="例如: 192.168.1.100")
                server_port = st.number_input("SSH端口", min_value=1, max_value=65535, value=22)
                server_username = st.text_input("用户名", placeholder="例如: ubuntu")
                
                auth_type = st.selectbox("认证方式", ["密码认证", "SSH密钥认证"])
                
                if auth_type == "密码认证":
                    server_password = st.text_input("密码", type="password")
                    server_key_file = None
                else:
                    server_password = None
                    server_key_file = st.text_input("SSH密钥文件路径", placeholder="例如: ~/.ssh/id_rsa")
                
                remote_work_dir = st.text_input("远程工作目录", value="/tmp/battery_training")
                
                submitted = st.form_submit_button("添加服务器")
                
                if submitted:
                    if server_name and server_host and server_username:
                        new_server = {
                            "name": server_name,
                            "host": server_host,
                            "port": server_port,
                            "username": server_username,
                            "password": server_password,
                            "key_file": server_key_file,
                            "auth_type": auth_type,
                            "remote_work_dir": remote_work_dir
                        }
                        
                        # 验证配置
                        test_config = {"servers": [new_server]}
                        is_valid, error_msg = st.session_state.config_manager.validate_config(test_config)
                        
                        if is_valid:
                            # 添加到配置
                            st.session_state.server_configs["servers"].append(new_server)
                            
                            # 保存配置
                            if st.session_state.config_manager.save_config(st.session_state.server_configs):
                                st.success(f"服务器 '{server_name}' 添加成功！")
                                st.rerun()
                            else:
                                st.error("保存配置失败")
                        else:
                            st.error(f"配置验证失败: {error_msg}")
                    else:
                        st.error("请填写所有必需字段")
        
        with col2:
            st.markdown("#### 已配置的服务器")
            
            if st.session_state.server_configs.get("servers"):
                for i, server in enumerate(st.session_state.server_configs["servers"]):
                    with st.expander(f"🖥️ {server['name']}"):
                        st.write(f"**地址**: {server['host']}:{server['port']}")
                        st.write(f"**用户名**: {server['username']}")
                        st.write(f"**认证方式**: {server.get('auth_type', '密码认证')}")
                        st.write(f"**工作目录**: {server.get('remote_work_dir', '/tmp')}")
                        
                        col_edit, col_delete = st.columns(2)
                        
                        with col_edit:
                            if st.button(f"设为默认", key=f"default_{i}"):
                                st.session_state.server_configs["default_server"] = server["name"]
                                st.session_state.config_manager.save_config(st.session_state.server_configs)
                                st.success(f"已设置 '{server['name']}' 为默认服务器")
                                st.rerun()
                        
                        with col_delete:
                            if st.button(f"删除", key=f"delete_{i}"):
                                st.session_state.server_configs["servers"].pop(i)
                                st.session_state.config_manager.save_config(st.session_state.server_configs)
                                st.success(f"已删除服务器 '{server['name']}'")
                                st.rerun()
            else:
                st.info("暂无配置的服务器")
    
    with tab2:
        st.subheader("连接管理")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 连接到服务器")
            
            if st.session_state.server_configs.get("servers"):
                server_names = [s["name"] for s in st.session_state.server_configs["servers"]]
                default_index = 0
                
                if st.session_state.server_configs.get("default_server"):
                    try:
                        default_index = server_names.index(st.session_state.server_configs["default_server"])
                    except ValueError:
                        pass
                
                selected_server_name = st.selectbox(
                    "选择服务器", 
                    server_names,
                    index=default_index
                )
                
                # 获取选中的服务器配置
                selected_server = None
                for server in st.session_state.server_configs["servers"]:
                    if server["name"] == selected_server_name:
                        selected_server = server
                        break
                
                if selected_server:
                    st.write(f"**服务器地址**: {selected_server['host']}:{selected_server['port']}")
                    st.write(f"**用户名**: {selected_server['username']}")
                    
                    col_connect, col_disconnect = st.columns(2)
                    
                    with col_connect:
                        if st.button("连接", disabled=st.session_state.server_manager.is_connected):
                            with st.spinner("正在连接..."):
                                success = st.session_state.server_manager.connect(
                                    host=selected_server["host"],
                                    port=selected_server["port"],
                                    username=selected_server["username"],
                                    password=selected_server.get("password"),
                                    key_file=selected_server.get("key_file")
                                )
                                
                                if success:
                                    st.success("连接成功！")
                                    st.rerun()
                                else:
                                    st.error("连接失败，请检查配置和网络")
                    
                    with col_disconnect:
                        if st.button("断开连接", disabled=not st.session_state.server_manager.is_connected):
                            st.session_state.server_manager.disconnect()
                            st.success("已断开连接")
                            st.rerun()
            else:
                st.warning("请先在'服务器配置'标签页中添加服务器")
        
        with col2:
            st.markdown("#### 连接状态")
            
            if st.session_state.server_manager.is_connected:
                # 测试连接
                if st.button("测试连接"):
                    with st.spinner("测试中..."):
                        test_result = st.session_state.server_manager.test_connection()
                        
                        if test_result["status"] == "connected":
                            st.success(f"✅ {test_result['message']}")
                            st.info(f"延迟: {test_result.get('latency', 'N/A')}")
                        else:
                            st.error(f"❌ {test_result['message']}")
                
                # 获取服务器信息
                if st.button("获取服务器信息"):
                    with st.spinner("获取信息中..."):
                        server_info = st.session_state.server_manager.get_server_info()
                        
                        if server_info:
                            st.markdown("**服务器信息**:")
                            for key, value in server_info.items():
                                st.write(f"- **{key}**: {value}")
                        else:
                            st.warning("无法获取服务器信息")
            else:
                st.info("未连接到服务器")
    
    with tab3:
        st.subheader("远程训练")
        
        if not st.session_state.server_manager.is_connected:
            st.warning("请先连接到服务器")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 数据上传")
                
                # 选择要上传的数据
                data_source = st.selectbox(
                    "数据源",
                    ["当前会话数据", "上传新文件"]
                )
                
                if data_source == "当前会话数据":
                    if st.session_state.features is not None:
                        st.info(f"将上传当前特征数据 ({st.session_state.features.shape[0]} 行, {st.session_state.features.shape[1]} 列)")
                        
                        if st.button("上传特征数据"):
                            with st.spinner("上传中..."):
                                # 保存特征数据到临时文件
                                temp_file = os.path.join(OUTPUT_FOLDER, "features_for_remote.csv")
                                st.session_state.features.to_csv(temp_file, index=False)
                                
                                # 上传到服务器
                                remote_path = "/tmp/battery_training/features.csv"
                                success = st.session_state.file_transfer.upload_file(temp_file, remote_path)
                                
                                if success:
                                    st.success("特征数据上传成功！")
                                else:
                                    st.error("上传失败")
                    else:
                        st.warning("当前会话中没有特征数据，请先完成特征提取")
                
                elif data_source == "上传新文件":
                    uploaded_file = st.file_uploader("选择数据文件", type=["csv", "xlsx"])
                    
                    if uploaded_file is not None:
                        if st.button("上传文件到服务器"):
                            with st.spinner("上传中..."):
                                # 保存上传的文件
                                temp_file = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
                                with open(temp_file, "wb") as f:
                                    f.write(uploaded_file.getbuffer())
                                
                                # 上传到服务器
                                remote_path = f"/tmp/battery_training/{uploaded_file.name}"
                                success = st.session_state.file_transfer.upload_file(temp_file, remote_path)
                                
                                if success:
                                    st.success(f"文件 {uploaded_file.name} 上传成功！")
                                else:
                                    st.error("上传失败")
            
            with col2:
                st.markdown("#### 训练配置")
                
                # 训练参数配置
                algorithm = st.selectbox(
                    "选择算法",
                    ["XGBoost", "RandomForest", "SVR", "LightGBM"]
                )
                
                # 根据算法显示不同参数
                if algorithm == "XGBoost":
                    n_estimators = st.slider("树的数量", 10, 200, 100, 10)
                    learning_rate = st.slider("学习率", 0.01, 0.3, 0.1, 0.01)
                    max_depth = st.slider("最大深度", 3, 10, 6, 1)
                    model_params = {
                        "n_estimators": n_estimators,
                        "learning_rate": learning_rate,
                        "max_depth": max_depth
                    }
                elif algorithm == "RandomForest":
                    n_estimators = st.slider("树的数量", 10, 200, 100, 10)
                    max_depth = st.slider("最大深度", 3, 20, 10, 1)
                    model_params = {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth
                    }
                elif algorithm == "SVR":
                    kernel = st.selectbox("核函数", ["rbf", "linear", "poly"])
                    C = st.slider("正则化参数 C", 0.1, 10.0, 1.0, 0.1)
                    model_params = {"kernel": kernel, "C": C}
                else:  # LightGBM
                    n_estimators = st.slider("树的数量", 10, 200, 100, 10)
                    learning_rate = st.slider("学习率", 0.01, 0.3, 0.1, 0.01)
                    model_params = {
                        "n_estimators": n_estimators,
                        "learning_rate": learning_rate
                    }
                
                # 其他配置
                target_col = st.selectbox("目标列", ["SOH", "capacity_max"])
                
                if st.session_state.features is not None:
                    feature_cols = st.multiselect(
                        "特征列",
                        [col for col in st.session_state.features.columns if col not in [target_col, 'cycle']],
                        default=[col for col in st.session_state.features.columns if col not in [target_col, 'cycle']][:5]
                    )
                else:
                    feature_cols = []
                
                # 启动训练
                if st.button("启动远程训练"):
                    if not feature_cols:
                        st.error("请选择特征列")
                    else:
                        with st.spinner("启动训练任务..."):
                            training_config = {
                                "algorithm": algorithm,
                                "model_params": model_params,
                                "target_col": target_col,
                                "feature_cols": feature_cols,
                                "data_file": "/tmp/battery_training/features.csv"
                            }
                            
                            job_id = st.session_state.remote_executor.start_training_job(training_config)
                            
                            if job_id:
                                st.success(f"训练任务已启动！任务ID: {job_id}")
                                st.session_state.current_job_id = job_id
                            else:
                                st.error("启动训练任务失败")
    
    with tab4:
        st.subheader("任务监控")
        
        if not st.session_state.server_manager.is_connected:
            st.warning("请先连接到服务器")
        else:
            # 任务状态监控
            if hasattr(st.session_state, 'current_job_id'):
                job_id = st.session_state.current_job_id
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"#### 任务状态: {job_id}")
                    
                    if st.button("刷新状态"):
                        with st.spinner("获取任务状态..."):
                            status = st.session_state.remote_executor.monitor_job_status(job_id)
                            
                            if status["status"] == "running":
                                st.info(f"🟡 {status['message']}")
                                st.write(f"运行时间: {status['running_time']:.1f} 秒")
                            elif status["status"] == "completed":
                                st.success(f"🟢 {status['message']}")
                            else:
                                st.error(f"🔴 {status['message']}")
                            
                            # 显示日志
                            if status.get("log"):
                                st.text_area("最新日志", status["log"], height=200)
                
                with col2:
                    st.markdown("#### 结果下载")
                    
                    if st.button("下载训练结果"):
                        with st.spinner("下载结果..."):
                            # 下载训练结果
                            local_results = os.path.join(OUTPUT_FOLDER, "remote_training_results.json")
                            success = st.session_state.file_transfer.download_file(
                                "/tmp/training_results.json", local_results
                            )
                            
                            if success:
                                # 显示结果
                                with open(local_results, 'r') as f:
                                    results = json.load(f)
                                
                                st.success("结果下载成功！")
                                st.json(results)
                                
                                # 提供下载链接
                                with open(local_results, "rb") as file:
                                    st.download_button(
                                        label="下载结果文件",
                                        data=file,
                                        file_name="remote_training_results.json",
                                        mime="application/json"
                                    )
                            else:
                                st.error("下载失败")
                    
                    if st.button("下载训练模型"):
                        with st.spinner("下载模型..."):
                            # 下载训练好的模型
                            local_model = os.path.join(MODELS_FOLDER, "remote_trained_model.pkl")
                            success = st.session_state.file_transfer.download_file(
                                "/tmp/trained_model.pkl", local_model
                            )
                            
                            if success:
                                st.success("模型下载成功！")
                                
                                # 提供下载链接
                                with open(local_model, "rb") as file:
                                    st.download_button(
                                        label="下载模型文件",
                                        data=file,
                                        file_name="remote_trained_model.pkl",
                                        mime="application/octet-stream"
                                    )
                            else:
                                st.error("下载失败")
            else:
                st.info("暂无运行中的任务")
                
                # 手动输入任务ID进行监控
                manual_job_id = st.text_input("输入任务ID进行监控")
                if manual_job_id and st.button("监控任务"):
                    st.session_state.current_job_id = manual_job_id
                    st.rerun()
    
    # 导航按钮
    if st.button("返回模型优化"):
        st.session_state.current_step = 7
        st.rerun()

# 页脚
st.markdown("---")
st.markdown("### 电池寿命预测系统 | 基于机器学习的SOH和RUL预测")
st.markdown("© 2025 电池健康管理团队")
st.markdown("© 2025 浙江锋锂新能源科技有限公司-唐光盛团队")

