# coding: utf-8
"""
电池数据预处理管道
提供电池数据的清洗、预处理和标准化功能
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from scipy import stats
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BatteryDataPreprocessor:
    """电池数据预处理器"""
    
    def __init__(self, data: pd.DataFrame):
        """
        初始化预处理器
        
        Args:
            data: 原始电池数据
        """
        self.original_data = data.copy()
        self.processed_data = None
        self.scaler = None
        self.imputer = None
        self.missing_values_filled = 0
        self.outliers_removed = 0
        
    def preprocess_data(self, cycle_col: str, voltage_col: str, current_col: str, 
                       time_col: str, capacity_col: str = None, temp_col: str = None,
                       remove_outliers: bool = True, fill_missing: bool = True,
                       normalize: bool = True, outlier_threshold: float = 3.0):
        """
        执行完整的数据预处理流程
        
        Args:
            cycle_col: 循环次数列名
            voltage_col: 电压列名
            current_col: 电流列名
            time_col: 时间列名
            capacity_col: 容量列名（可选）
            temp_col: 温度列名（可选）
            remove_outliers: 是否移除异常值
            fill_missing: 是否填充缺失值
            normalize: 是否标准化数据
            outlier_threshold: 异常值阈值（标准差倍数）
        """
        logger.info("开始数据预处理...")
        
        # 复制数据
        self.processed_data = self.original_data.copy()
        
        # 1. 数据类型转换
        self._convert_data_types(cycle_col, voltage_col, current_col, time_col, 
                                capacity_col, temp_col)
        
        # 2. 处理缺失值
        if fill_missing:
            self._handle_missing_values()
        
        # 3. 移除异常值
        if remove_outliers:
            self._remove_outliers(outlier_threshold)
        
        # 4. 数据标准化
        if normalize:
            self._normalize_data(cycle_col, voltage_col, current_col, time_col,
                               capacity_col, temp_col)
        
        # 5. 计算SOH（如果有容量数据）
        if capacity_col:
            self._calculate_soh(cycle_col, capacity_col)
        
        logger.info(f"数据预处理完成。原始数据: {self.original_data.shape[0]} 行，"
                   f"处理后: {self.processed_data.shape[0]} 行")
        
    def _convert_data_types(self, cycle_col: str, voltage_col: str, current_col: str,
                           time_col: str, capacity_col: str = None, temp_col: str = None):
        """转换数据类型"""
        try:
            # 确保数值列为数值类型
            numeric_cols = [cycle_col, voltage_col, current_col]
            if capacity_col:
                numeric_cols.append(capacity_col)
            if temp_col:
                numeric_cols.append(temp_col)
            
            for col in numeric_cols:
                if col in self.processed_data.columns:
                    self.processed_data[col] = pd.to_numeric(self.processed_data[col], errors='coerce')
            
            # 处理时间列
            if time_col in self.processed_data.columns:
                try:
                    self.processed_data[time_col] = pd.to_datetime(self.processed_data[time_col])
                except:
                    # 如果时间列是数值型（如秒数），保持原样
                    self.processed_data[time_col] = pd.to_numeric(self.processed_data[time_col], errors='coerce')
            
            logger.info("数据类型转换完成")
            
        except Exception as e:
            logger.error(f"数据类型转换失败: {str(e)}")
            raise
    
    def _handle_missing_values(self):
        """处理缺失值"""
        try:
            # 统计缺失值
            missing_before = self.processed_data.isnull().sum().sum()
            
            if missing_before > 0:
                # 对数值列使用均值填充
                numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    self.imputer = SimpleImputer(strategy='mean')
                    self.processed_data[numeric_cols] = self.imputer.fit_transform(
                        self.processed_data[numeric_cols]
                    )
                
                # 对分类列使用众数填充
                categorical_cols = self.processed_data.select_dtypes(include=['object']).columns
                for col in categorical_cols:
                    mode_value = self.processed_data[col].mode()
                    if len(mode_value) > 0:
                        self.processed_data[col].fillna(mode_value[0], inplace=True)
                
                missing_after = self.processed_data.isnull().sum().sum()
                self.missing_values_filled = missing_before - missing_after
                
                logger.info(f"缺失值处理完成，填充了 {self.missing_values_filled} 个缺失值")
            
        except Exception as e:
            logger.error(f"处理缺失值失败: {str(e)}")
            raise
    
    def _remove_outliers(self, threshold: float = 3.0):
        """移除异常值"""
        try:
            original_count = len(self.processed_data)
            
            # 对数值列检测异常值
            numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                # 使用Z-score方法检测异常值
                z_scores = np.abs(stats.zscore(self.processed_data[col]))
                outlier_mask = z_scores > threshold
                
                # 移除异常值
                self.processed_data = self.processed_data[~outlier_mask]
            
            self.outliers_removed = original_count - len(self.processed_data)
            
            logger.info(f"异常值移除完成，移除了 {self.outliers_removed} 行数据")
            
        except Exception as e:
            logger.error(f"移除异常值失败: {str(e)}")
            raise
    
    def _normalize_data(self, cycle_col: str, voltage_col: str, current_col: str,
                       time_col: str, capacity_col: str = None, temp_col: str = None):
        """标准化数据"""
        try:
            # 选择需要标准化的列（排除循环次数）
            cols_to_normalize = [voltage_col, current_col]
            if capacity_col:
                cols_to_normalize.append(capacity_col)
            if temp_col:
                cols_to_normalize.append(temp_col)
            
            # 过滤存在的列
            cols_to_normalize = [col for col in cols_to_normalize 
                               if col in self.processed_data.columns]
            
            if cols_to_normalize:
                self.scaler = StandardScaler()
                self.processed_data[cols_to_normalize] = self.scaler.fit_transform(
                    self.processed_data[cols_to_normalize]
                )
                
                logger.info(f"数据标准化完成，标准化了 {len(cols_to_normalize)} 列")
            
        except Exception as e:
            logger.error(f"数据标准化失败: {str(e)}")
            raise
    
    def _calculate_soh(self, cycle_col: str, capacity_col: str):
        """计算健康状态(SOH)"""
        try:
            if capacity_col in self.processed_data.columns:
                # 获取初始容量（第一个循环的容量）
                initial_capacity = self.processed_data[self.processed_data[cycle_col] == 
                                                     self.processed_data[cycle_col].min()][capacity_col].mean()
                
                # 计算SOH = 当前容量 / 初始容量
                self.processed_data['SOH'] = self.processed_data[capacity_col] / initial_capacity
                
                # 确保SOH在合理范围内
                self.processed_data['SOH'] = self.processed_data['SOH'].clip(0, 1)
                
                logger.info("SOH计算完成")
            
        except Exception as e:
            logger.error(f"SOH计算失败: {str(e)}")
            raise
    
    def get_preprocessing_summary(self) -> dict:
        """获取预处理摘要"""
        return {
            "original_rows": self.original_data.shape[0],
            "processed_rows": self.processed_data.shape[0] if self.processed_data is not None else 0,
            "original_columns": self.original_data.shape[1],
            "processed_columns": self.processed_data.shape[1] if self.processed_data is not None else 0,
            "missing_values_filled": self.missing_values_filled,
            "outliers_removed": self.outliers_removed,
            "scaler_used": self.scaler is not None,
            "imputer_used": self.imputer is not None
        }
    
    def save_processed_data(self, filepath: str):
        """保存预处理后的数据"""
        if self.processed_data is not None:
            self.processed_data.to_csv(filepath, index=False)
            logger.info(f"预处理数据已保存到: {filepath}")
        else:
            logger.warning("没有预处理数据可保存")
    
    def load_processed_data(self, filepath: str):
        """加载预处理后的数据"""
        try:
            self.processed_data = pd.read_csv(filepath)
            logger.info(f"预处理数据已从 {filepath} 加载")
        except Exception as e:
            logger.error(f"加载预处理数据失败: {str(e)}")
            raise

