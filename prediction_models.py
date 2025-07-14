# coding: utf-8
"""
电池预测模型模块
提供多种机器学习模型用于电池寿命预测
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
import warnings
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class BatteryPredictionModel:
    """电池预测模型"""
    
    def __init__(self):
        """初始化预测模型"""
        self.model = None
        self.model_type = None
        self.scaler = None
        self.feature_cols = None
        self.target_col = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.training_history = None
        
    def train_model(self, data: pd.DataFrame, target_col: str, feature_cols: List[str],
                   model_type: str = 'XGBoost', model_params: Dict = None,
                   train_ratio: float = 0.8, random_state: int = 42):
        """
        训练预测模型
        
        Args:
            data: 训练数据
            target_col: 目标列名
            feature_cols: 特征列名列表
            model_type: 模型类型
            model_params: 模型参数
            train_ratio: 训练集比例
            random_state: 随机种子
        """
        try:
            logger.info(f"开始训练 {model_type} 模型...")
            
            # 保存配置
            self.model_type = model_type
            self.feature_cols = feature_cols
            self.target_col = target_col
            
            # 准备数据
            X = data[feature_cols].fillna(0)
            y = data[target_col].fillna(0)
            
            # 分割数据
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, train_size=train_ratio, random_state=random_state
            )
            
            # 数据标准化（除了树模型）
            if model_type not in ['RandomForest', 'XGBoost', 'LightGBM']:
                self.scaler = StandardScaler()
                self.X_train = self.scaler.fit_transform(self.X_train)
                self.X_test = self.scaler.transform(self.X_test)
            
            # 设置默认参数
            if model_params is None:
                model_params = {}
            
            # 训练模型
            if model_type == 'SVR':
                self.model = SVR(**model_params)
                self.model.fit(self.X_train, self.y_train)
                
            elif model_type == 'RandomForest':
                default_params = {'n_estimators': 100, 'max_depth': 10, 'random_state': random_state}
                default_params.update(model_params)
                self.model = RandomForestRegressor(**default_params)
                self.model.fit(self.X_train, self.y_train)
                
            elif model_type == 'XGBoost':
                default_params = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6, 'random_state': random_state}
                default_params.update(model_params)
                self.model = xgb.XGBRegressor(**default_params)
                self.model.fit(self.X_train, self.y_train)
                
            elif model_type == 'LightGBM':
                default_params = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6, 'random_state': random_state}
                default_params.update(model_params)
                self.model = lgb.LGBMRegressor(**default_params)
                self.model.fit(self.X_train, self.y_train)
                
            elif model_type == 'LSTM':
                self._train_lstm_model(model_params)
                
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
            
            # 预测
            self.y_pred = self.predict(self.X_test)
            
            logger.info(f"{model_type} 模型训练完成")
            
        except Exception as e:
            logger.error(f"模型训练失败: {str(e)}")
            raise
    
    def _train_lstm_model(self, model_params: Dict):
        """训练LSTM模型"""
        try:
            # LSTM参数
            units = model_params.get('units', 64)
            epochs = model_params.get('epochs', 50)
            batch_size = model_params.get('batch_size', 32)
            
            # 重塑数据为LSTM格式 (samples, timesteps, features)
            # 这里简化处理，将每个样本作为一个时间步
            X_train_lstm = self.X_train.reshape((self.X_train.shape[0], 1, self.X_train.shape[1]))
            X_test_lstm = self.X_test.reshape((self.X_test.shape[0], 1, self.X_test.shape[1]))
            
            # 构建LSTM模型
            self.model = Sequential([
                LSTM(units, return_sequences=True, input_shape=(1, self.X_train.shape[1])),
                Dropout(0.2),
                LSTM(units // 2, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
            
            # 训练模型
            history = self.model.fit(
                X_train_lstm, self.y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_test_lstm, self.y_test),
                verbose=0
            )
            
            self.training_history = history.history
            
            # 更新测试数据格式
            self.X_test = X_test_lstm
            
        except Exception as e:
            logger.error(f"LSTM模型训练失败: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        模型预测
        
        Args:
            X: 输入特征
            
        Returns:
            np.ndarray: 预测结果
        """
        try:
            if self.model is None:
                raise ValueError("模型未训练")
            
            if self.model_type == 'LSTM':
                if len(X.shape) == 2:
                    X = X.reshape((X.shape[0], 1, X.shape[1]))
                predictions = self.model.predict(X, verbose=0).flatten()
            else:
                predictions = self.model.predict(X)
            
            return predictions
            
        except Exception as e:
            logger.error(f"预测失败: {str(e)}")
            raise
    
    def evaluate_model(self) -> Dict[str, float]:
        """
        评估模型性能
        
        Returns:
            Dict[str, float]: 评估指标
        """
        try:
            if self.y_pred is None:
                raise ValueError("模型未预测")
            
            metrics = {
                'mse': mean_squared_error(self.y_test, self.y_pred),
                'rmse': np.sqrt(mean_squared_error(self.y_test, self.y_pred)),
                'mae': mean_absolute_error(self.y_test, self.y_pred),
                'r2': r2_score(self.y_test, self.y_pred)
            }
            
            # 计算MAPE
            mape = np.mean(np.abs((self.y_test - self.y_pred) / self.y_test)) * 100
            metrics['mape'] = mape
            
            logger.info(f"模型评估完成: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"模型评估失败: {str(e)}")
            raise
    
    def plot_prediction_vs_actual(self, figsize: tuple = (10, 6)) -> plt.Figure:
        """
        绘制预测值vs实际值图
        
        Args:
            figsize: 图形大小
            
        Returns:
            matplotlib.Figure: 图形对象
        """
        try:
            if self.y_pred is None:
                raise ValueError("模型未预测")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            
            # 散点图
            ax1.scatter(self.y_test, self.y_pred, alpha=0.6, s=50)
            
            # 完美预测线
            min_val = min(self.y_test.min(), self.y_pred.min())
            max_val = max(self.y_test.max(), self.y_pred.max())
            ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测')
            
            ax1.set_xlabel('实际值')
            ax1.set_ylabel('预测值')
            ax1.set_title('预测值 vs 实际值')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 残差图
            residuals = self.y_test - self.y_pred
            ax2.scatter(self.y_pred, residuals, alpha=0.6, s=50)
            ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
            ax2.set_xlabel('预测值')
            ax2.set_ylabel('残差')
            ax2.set_title('残差图')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            logger.error(f"绘制预测图失败: {str(e)}")
            raise
    
    def plot_feature_importance(self, figsize: tuple = (12, 8)) -> plt.Figure:
        """
        绘制特征重要性图
        
        Args:
            figsize: 图形大小
            
        Returns:
            matplotlib.Figure: 图形对象
        """
        try:
            if self.model is None:
                raise ValueError("模型未训练")
            
            # 获取特征重要性
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
            elif hasattr(self.model, 'coef_'):
                importances = np.abs(self.model.coef_)
            else:
                logger.warning("模型不支持特征重要性分析")
                return None
            
            # 创建特征重要性DataFrame
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # 绘制图形
            fig, ax = plt.subplots(figsize=figsize)
            
            # 选择前20个最重要的特征
            top_features = feature_importance_df.head(20)
            
            bars = ax.barh(range(len(top_features)), top_features['importance'])
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'])
            ax.set_xlabel('重要性')
            ax.set_title(f'{self.model_type} 模型特征重要性')
            ax.grid(True, alpha=0.3)
            
            # 添加数值标签
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2, 
                       f'{width:.3f}', ha='left', va='center')
            
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            logger.error(f"绘制特征重要性图失败: {str(e)}")
            raise
    
    def predict_future(self, cycles_to_predict: int, prediction_method: str = 'direct',
                      confidence_level: float = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        预测未来SOH
        
        Args:
            cycles_to_predict: 要预测的循环数
            prediction_method: 预测方法
            confidence_level: 置信水平
            
        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: (预测值, 置信区间)
        """
        try:
            if self.model is None:
                raise ValueError("模型未训练")
            
            # 获取最后一个样本的特征
            last_features = self.X_test[-1:] if len(self.X_test) > 0 else self.X_train[-1:]
            
            predictions = []
            confidence_intervals = []
            
            for i in range(cycles_to_predict):
                # 预测下一个值
                if self.model_type == 'LSTM':
                    if len(last_features.shape) == 2:
                        last_features_lstm = last_features.reshape((1, 1, last_features.shape[1]))
                    else:
                        last_features_lstm = last_features
                    pred = self.model.predict(last_features_lstm, verbose=0)[0, 0]
                else:
                    pred = self.model.predict(last_features)[0]
                
                predictions.append(pred)
                
                # 计算置信区间（简化方法）
                if confidence_level:
                    # 基于训练误差估计置信区间
                    train_error = np.std(self.y_train - self.predict(self.X_train))
                    z_score = 1.96 if confidence_level == 0.95 else 2.58  # 95% or 99%
                    confidence_interval = z_score * train_error
                    confidence_intervals.append(confidence_interval)
                
                # 更新特征（简化方法：假设其他特征保持不变，只更新SOH相关特征）
                if prediction_method == 'recursive':
                    # 递归预测：使用预测值更新特征
                    # 这里需要根据具体的特征工程逻辑来实现
                    pass
            
            predictions = np.array(predictions)
            confidence_intervals = np.array(confidence_intervals) if confidence_intervals else None
            
            logger.info(f"未来 {cycles_to_predict} 个循环的预测完成")
            return predictions, confidence_intervals
            
        except Exception as e:
            logger.error(f"未来预测失败: {str(e)}")
            raise
    
    def calculate_rul(self, predictions: np.ndarray, eol_threshold: float = 0.8) -> int:
        """
        计算剩余使用寿命(RUL)
        
        Args:
            predictions: 预测的SOH值
            eol_threshold: EOL阈值
            
        Returns:
            int: 剩余使用寿命（循环数）
        """
        try:
            # 找到第一个低于EOL阈值的点
            below_threshold = predictions < eol_threshold
            
            if np.any(below_threshold):
                rul = np.argmax(below_threshold)
            else:
                rul = len(predictions)  # 在预测范围内未达到EOL
            
            logger.info(f"计算得到RUL: {rul} 循环")
            return rul
            
        except Exception as e:
            logger.error(f"RUL计算失败: {str(e)}")
            raise
    
    def plot_predictions(self, predictions: np.ndarray, confidence: np.ndarray = None,
                        eol_threshold: float = 0.8, figsize: tuple = (12, 6)) -> plt.Figure:
        """
        绘制预测结果图
        
        Args:
            predictions: 预测值
            confidence: 置信区间
            eol_threshold: EOL阈值
            figsize: 图形大小
            
        Returns:
            matplotlib.Figure: 图形对象
        """
        try:
            fig, ax = plt.subplots(figsize=figsize)
            
            # 历史数据（如果有）
            if hasattr(self, 'y_test') and self.y_test is not None:
                historical_cycles = range(len(self.y_test))
                ax.plot(historical_cycles, self.y_test, 'b-', linewidth=2, label='历史数据', alpha=0.8)
            
            # 预测数据
            future_cycles = range(len(self.y_test) if hasattr(self, 'y_test') and self.y_test is not None else 0, 
                                len(self.y_test) + len(predictions) if hasattr(self, 'y_test') and self.y_test is not None else len(predictions))
            ax.plot(future_cycles, predictions, 'r--', linewidth=2, label='预测值', alpha=0.8)
            
            # 置信区间
            if confidence is not None:
                ax.fill_between(future_cycles, 
                              predictions - confidence, 
                              predictions + confidence,
                              alpha=0.3, color='red', label='置信区间')
            
            # EOL阈值线
            ax.axhline(y=eol_threshold, color='orange', linestyle=':', linewidth=2, 
                      label=f'EOL阈值 ({eol_threshold})')
            
            # RUL标记
            rul = self.calculate_rul(predictions, eol_threshold)
            if rul < len(predictions):
                rul_cycle = future_cycles[rul]
                ax.axvline(x=rul_cycle, color='red', linestyle=':', linewidth=2, 
                          label=f'预测EOL (循环 {rul_cycle})')
            
            ax.set_xlabel('循环次数')
            ax.set_ylabel('SOH')
            ax.set_title('电池SOH预测')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.1)
            
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            logger.error(f"绘制预测图失败: {str(e)}")
            raise
    
    def plot_rul(self, predictions: np.ndarray, eol_threshold: float = 0.8,
                figsize: tuple = (10, 6)) -> plt.Figure:
        """
        绘制RUL预测图
        
        Args:
            predictions: 预测值
            eol_threshold: EOL阈值
            figsize: 图形大小
            
        Returns:
            matplotlib.Figure: 图形对象
        """
        try:
            fig, ax = plt.subplots(figsize=figsize)
            
            # 计算每个时间点的RUL
            rul_values = []
            for i in range(len(predictions)):
                remaining_predictions = predictions[i:]
                rul = self.calculate_rul(remaining_predictions, eol_threshold)
                rul_values.append(rul)
            
            cycles = range(len(predictions))
            ax.plot(cycles, rul_values, 'g-', linewidth=2, marker='o', markersize=4)
            
            ax.set_xlabel('循环次数')
            ax.set_ylabel('剩余使用寿命 (RUL)')
            ax.set_title('电池剩余使用寿命预测')
            ax.grid(True, alpha=0.3)
            
            # 添加趋势线
            if len(rul_values) > 1:
                z = np.polyfit(cycles, rul_values, 1)
                p = np.poly1d(z)
                ax.plot(cycles, p(cycles), "r--", alpha=0.8, 
                       label=f'趋势线 (斜率: {z[0]:.2f})')
                ax.legend()
            
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            logger.error(f"绘制RUL图失败: {str(e)}")
            raise
    
    def save_model(self, filepath: str):
        """保存模型"""
        try:
            model_data = {
                'model': self.model,
                'model_type': self.model_type,
                'scaler': self.scaler,
                'feature_cols': self.feature_cols,
                'target_col': self.target_col,
                'training_history': self.training_history
            }
            
            if self.model_type == 'LSTM':
                # Keras模型需要特殊处理
                self.model.save(filepath.replace('.pkl', '.h5'))
                model_data['model'] = None  # 不保存在pkl中
            
            joblib.dump(model_data, filepath)
            logger.info(f"模型已保存到: {filepath}")
            
        except Exception as e:
            logger.error(f"保存模型失败: {str(e)}")
            raise
    
    def load_model(self, filepath: str):
        """加载模型"""
        try:
            model_data = joblib.load(filepath)
            
            self.model_type = model_data['model_type']
            self.scaler = model_data['scaler']
            self.feature_cols = model_data['feature_cols']
            self.target_col = model_data['target_col']
            self.training_history = model_data.get('training_history')
            
            if self.model_type == 'LSTM':
                from tensorflow.keras.models import load_model
                self.model = load_model(filepath.replace('.pkl', '.h5'))
            else:
                self.model = model_data['model']
            
            logger.info(f"模型已从 {filepath} 加载")
            
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            raise

