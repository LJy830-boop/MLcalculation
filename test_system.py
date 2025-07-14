# coding: utf-8
"""
电池寿命预测系统测试脚本
用于验证各个模块的功能是否正常
"""

import os
import sys
import warnings
import traceback
import pandas as pd
import numpy as np
from typing import Dict, List

warnings.filterwarnings('ignore')

def test_imports():
    """测试模块导入"""
    print("🔍 测试模块导入...")
    
    try:
        # 测试核心依赖
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import sklearn
        print("✅ 核心依赖导入成功")
        
        # 测试机器学习库
        import xgboost as xgb
        import lightgbm as lgb
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.svm import SVR
        print("✅ 机器学习库导入成功")
        
        # 测试深度学习库
        try:
            import tensorflow as tf
            print("✅ TensorFlow导入成功")
        except ImportError:
            print("⚠️ TensorFlow导入失败，LSTM功能可能不可用")
        
        # 测试SSH连接库
        try:
            import paramiko
            from scp import SCPClient
            print("✅ SSH连接库导入成功")
        except ImportError:
            print("⚠️ SSH连接库导入失败，服务器连接功能不可用")
        
        # 测试自定义模块
        try:
            from data_preprocessing_pipeline import BatteryDataPreprocessor
            from exploratory_data_analysis import BatteryDataExplorer
            from feature_extraction import BatteryFeatureExtractor
            from prediction_models import BatteryPredictionModel
            from model_evaluation import ModelEvaluator
            from server_connection import ServerConnectionManager
            print("✅ 自定义模块导入成功")
        except ImportError as e:
            print(f"❌ 自定义模块导入失败: {str(e)}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 模块导入测试失败: {str(e)}")
        return False

def create_test_data():
    """创建测试数据"""
    print("📊 创建测试数据...")
    
    try:
        # 生成模拟电池数据
        np.random.seed(42)
        
        cycles = []
        voltages = []
        currents = []
        times = []
        capacities = []
        temperatures = []
        
        initial_capacity = 2.5
        
        for cycle in range(1, 101):  # 100个循环
            # 每个循环生成20个数据点
            cycle_data_points = 20
            
            for point in range(cycle_data_points):
                cycles.append(cycle)
                
                # 电压：随循环衰减，加入噪声
                base_voltage = 3.2 + 0.8 * (point / cycle_data_points) - 0.001 * cycle
                voltage = base_voltage + np.random.normal(0, 0.05)
                voltages.append(max(2.5, min(4.2, voltage)))
                
                # 电流：充放电过程
                current = 1.5 * np.sin(2 * np.pi * point / cycle_data_points) + np.random.normal(0, 0.1)
                currents.append(current)
                
                # 时间
                times.append(point * 60)  # 每分钟一个数据点
                
                # 容量：随循环衰减
                capacity_fade = 0.002 * cycle + np.random.normal(0, 0.01)
                capacity = initial_capacity * (1 - capacity_fade)
                capacities.append(max(1.0, capacity))
                
                # 温度
                temperature = 25 + np.random.normal(0, 2)
                temperatures.append(temperature)
        
        # 创建DataFrame
        test_data = pd.DataFrame({
            'cycle': cycles,
            'voltage': voltages,
            'current': currents,
            'time': times,
            'capacity': capacities,
            'temperature': temperatures
        })
        
        print(f"✅ 测试数据创建成功，形状: {test_data.shape}")
        return test_data
        
    except Exception as e:
        print(f"❌ 测试数据创建失败: {str(e)}")
        return None

def test_data_preprocessing(data):
    """测试数据预处理"""
    print("🔧 测试数据预处理...")
    
    try:
        from data_preprocessing_pipeline import BatteryDataPreprocessor
        
        preprocessor = BatteryDataPreprocessor(data)
        preprocessor.preprocess_data(
            cycle_col='cycle',
            voltage_col='voltage',
            current_col='current',
            time_col='time',
            capacity_col='capacity',
            temp_col='temperature'
        )
        
        if preprocessor.processed_data is not None:
            print(f"✅ 数据预处理成功，处理后形状: {preprocessor.processed_data.shape}")
            return preprocessor.processed_data
        else:
            print("❌ 数据预处理失败")
            return None
            
    except Exception as e:
        print(f"❌ 数据预处理测试失败: {str(e)}")
        traceback.print_exc()
        return None

def test_feature_extraction(data):
    """测试特征提取"""
    print("🎯 测试特征提取...")
    
    try:
        from feature_extraction import BatteryFeatureExtractor
        
        extractor = BatteryFeatureExtractor(data)
        
        # 提取时域特征
        time_features = extractor.extract_time_domain_features(
            cycle_col='cycle',
            voltage_col='voltage',
            current_col='current',
            time_col='time',
            capacity_col='capacity'
        )
        
        if time_features is not None and not time_features.empty:
            print(f"✅ 时域特征提取成功，特征数: {len(time_features.columns)-1}")
        else:
            print("❌ 时域特征提取失败")
            return None
        
        # 计算SOH特征
        soh_features = extractor.calculate_soh_features(
            cycle_col='cycle',
            capacity_col='capacity'
        )
        
        if soh_features is not None and not soh_features.empty:
            print(f"✅ SOH特征计算成功")
        else:
            print("❌ SOH特征计算失败")
        
        return extractor.features
        
    except Exception as e:
        print(f"❌ 特征提取测试失败: {str(e)}")
        traceback.print_exc()
        return None

def test_model_training(features):
    """测试模型训练"""
    print("🤖 测试模型训练...")
    
    try:
        from prediction_models import BatteryPredictionModel
        
        if 'SOH' not in features.columns:
            print("❌ 特征数据中缺少SOH列")
            return None
        
        # 选择特征列
        feature_cols = [col for col in features.columns 
                       if col not in ['cycle', 'SOH'] and 
                       features[col].dtype in [np.float64, np.int64]]
        
        if len(feature_cols) < 5:
            print(f"❌ 可用特征数量不足: {len(feature_cols)}")
            return None
        
        # 使用前10个特征进行测试
        feature_cols = feature_cols[:10]
        
        model = BatteryPredictionModel()
        
        # 测试XGBoost模型
        model.train_model(
            data=features,
            target_col='SOH',
            feature_cols=feature_cols,
            model_type='XGBoost',
            model_params={'n_estimators': 10, 'max_depth': 3},  # 减少参数以加快测试
            train_ratio=0.8
        )
        
        if model.model is not None:
            print("✅ XGBoost模型训练成功")
            
            # 评估模型
            metrics = model.evaluate_model()
            print(f"   R²: {metrics['r2']:.4f}")
            print(f"   RMSE: {metrics['rmse']:.4f}")
            
            return model
        else:
            print("❌ 模型训练失败")
            return None
            
    except Exception as e:
        print(f"❌ 模型训练测试失败: {str(e)}")
        traceback.print_exc()
        return None

def test_server_connection():
    """测试服务器连接模块"""
    print("🌐 测试服务器连接模块...")
    
    try:
        from server_connection import (
            ServerConnectionManager, 
            FileTransferManager, 
            RemoteCommandExecutor, 
            ServerConfigManager
        )
        
        # 测试配置管理器
        config_manager = ServerConfigManager("test_config.json")
        test_config = {
            "servers": [
                {
                    "name": "测试服务器",
                    "host": "localhost",
                    "port": 22,
                    "username": "test",
                    "password": "test123"
                }
            ]
        }
        
        # 验证配置
        is_valid, error_msg = config_manager.validate_config(test_config)
        if is_valid:
            print("✅ 服务器配置验证成功")
        else:
            print(f"⚠️ 服务器配置验证失败: {error_msg}")
        
        # 测试连接管理器（不实际连接）
        connection_manager = ServerConnectionManager()
        print("✅ 服务器连接管理器初始化成功")
        
        # 测试文件传输管理器
        file_transfer = FileTransferManager(connection_manager)
        print("✅ 文件传输管理器初始化成功")
        
        # 测试远程执行器
        remote_executor = RemoteCommandExecutor(connection_manager)
        print("✅ 远程执行器初始化成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 服务器连接模块测试失败: {str(e)}")
        traceback.print_exc()
        return False

def test_streamlit_app():
    """测试Streamlit应用"""
    print("🎨 测试Streamlit应用...")
    
    try:
        # 检查主应用文件是否存在
        app_files = [
            'streamlit_app.py',
            'battery_prediction_app_with_server.py'
        ]
        
        for app_file in app_files:
            if os.path.exists(app_file):
                print(f"✅ 应用文件 {app_file} 存在")
            else:
                print(f"❌ 应用文件 {app_file} 不存在")
        
        # 检查配置文件
        config_files = [
            '.streamlit/config.toml',
            '.streamlit/secrets.toml',
            'requirements.txt',
            'packages.txt'
        ]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                print(f"✅ 配置文件 {config_file} 存在")
            else:
                print(f"⚠️ 配置文件 {config_file} 不存在")
        
        return True
        
    except Exception as e:
        print(f"❌ Streamlit应用测试失败: {str(e)}")
        return False

def run_all_tests():
    """运行所有测试"""
    print("🚀 开始系统测试...\n")
    
    test_results = {}
    
    # 1. 测试模块导入
    test_results['imports'] = test_imports()
    print()
    
    # 2. 创建测试数据
    test_data = create_test_data()
    test_results['test_data'] = test_data is not None
    print()
    
    if test_data is not None:
        # 3. 测试数据预处理
        processed_data = test_data_preprocessing(test_data)
        test_results['preprocessing'] = processed_data is not None
        print()
        
        if processed_data is not None:
            # 4. 测试特征提取
            features = test_feature_extraction(processed_data)
            test_results['feature_extraction'] = features is not None
            print()
            
            if features is not None:
                # 5. 测试模型训练
                model = test_model_training(features)
                test_results['model_training'] = model is not None
                print()
    
    # 6. 测试服务器连接
    test_results['server_connection'] = test_server_connection()
    print()
    
    # 7. 测试Streamlit应用
    test_results['streamlit_app'] = test_streamlit_app()
    print()
    
    # 汇总测试结果
    print("📋 测试结果汇总:")
    print("=" * 50)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    for test_name, result in test_results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:20} : {status}")
    
    print("=" * 50)
    print(f"总测试数: {total_tests}")
    print(f"通过数量: {passed_tests}")
    print(f"失败数量: {total_tests - passed_tests}")
    print(f"通过率: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 所有测试通过！系统准备就绪。")
    elif passed_tests >= total_tests * 0.8:
        print("\n⚠️ 大部分测试通过，系统基本可用。")
    else:
        print("\n❌ 多个测试失败，请检查系统配置。")
    
    return test_results

if __name__ == "__main__":
    run_all_tests()

