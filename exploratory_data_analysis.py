# coding: utf-8
"""
电池数据探索性分析模块
提供电池数据的可视化和统计分析功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class BatteryDataExplorer:
    """电池数据探索器"""
    
    def __init__(self, data: pd.DataFrame):
        """
        初始化数据探索器
        
        Args:
            data: 电池数据
        """
        self.data = data.copy()
        
    def plot_distributions(self, columns: List[str], figsize: tuple = (15, 10)) -> plt.Figure:
        """
        绘制数据分布图
        
        Args:
            columns: 要绘制的列名列表
            figsize: 图形大小
            
        Returns:
            matplotlib.Figure: 图形对象
        """
        try:
            n_cols = min(3, len(columns))
            n_rows = (len(columns) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(columns):
                if col in self.data.columns:
                    # 直方图
                    axes[i].hist(self.data[col].dropna(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                    axes[i].set_title(f'{col} 分布')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('频次')
                    axes[i].grid(True, alpha=0.3)
                    
                    # 添加统计信息
                    mean_val = self.data[col].mean()
                    std_val = self.data[col].std()
                    axes[i].axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'均值: {mean_val:.3f}')
                    axes[i].legend()
            
            # 隐藏多余的子图
            for i in range(len(columns), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            logger.info(f"数据分布图绘制完成，包含 {len(columns)} 个变量")
            return fig
            
        except Exception as e:
            logger.error(f"绘制分布图失败: {str(e)}")
            raise
    
    def plot_correlation_matrix(self, figsize: tuple = (12, 10)) -> plt.Figure:
        """
        绘制相关性矩阵热力图
        
        Args:
            figsize: 图形大小
            
        Returns:
            matplotlib.Figure: 图形对象
        """
        try:
            # 选择数值列
            numeric_data = self.data.select_dtypes(include=[np.number])
            
            if numeric_data.empty:
                raise ValueError("数据中没有数值列")
            
            # 计算相关性矩阵
            corr_matrix = numeric_data.corr()
            
            # 绘制热力图
            fig, ax = plt.subplots(figsize=figsize)
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
            
            ax.set_title('变量相关性矩阵', fontsize=16, pad=20)
            plt.tight_layout()
            
            logger.info("相关性矩阵绘制完成")
            return fig
            
        except Exception as e:
            logger.error(f"绘制相关性矩阵失败: {str(e)}")
            raise
    
    def plot_capacity_fade(self, cycle_col: str, capacity_col: str, 
                          figsize: tuple = (12, 6)) -> plt.Figure:
        """
        绘制容量退化曲线
        
        Args:
            cycle_col: 循环次数列名
            capacity_col: 容量列名
            figsize: 图形大小
            
        Returns:
            matplotlib.Figure: 图形对象
        """
        try:
            if cycle_col not in self.data.columns or capacity_col not in self.data.columns:
                raise ValueError(f"列 {cycle_col} 或 {capacity_col} 不存在")
            
            # 按循环次数分组，计算平均容量
            capacity_by_cycle = self.data.groupby(cycle_col)[capacity_col].mean().reset_index()
            
            fig, ax = plt.subplots(figsize=figsize)
            
            # 绘制容量退化曲线
            ax.plot(capacity_by_cycle[cycle_col], capacity_by_cycle[capacity_col], 
                   'b-', linewidth=2, marker='o', markersize=4, alpha=0.8)
            
            ax.set_xlabel('循环次数')
            ax.set_ylabel('容量')
            ax.set_title('电池容量退化曲线')
            ax.grid(True, alpha=0.3)
            
            # 添加趋势线
            z = np.polyfit(capacity_by_cycle[cycle_col], capacity_by_cycle[capacity_col], 1)
            p = np.poly1d(z)
            ax.plot(capacity_by_cycle[cycle_col], p(capacity_by_cycle[cycle_col]), 
                   "r--", alpha=0.8, label=f'趋势线 (斜率: {z[0]:.6f})')
            
            ax.legend()
            plt.tight_layout()
            
            logger.info("容量退化曲线绘制完成")
            return fig
            
        except Exception as e:
            logger.error(f"绘制容量退化曲线失败: {str(e)}")
            raise
    
    def plot_soh_curve(self, cycle_col: str, capacity_col: str, 
                      figsize: tuple = (12, 6)) -> plt.Figure:
        """
        绘制SOH曲线
        
        Args:
            cycle_col: 循环次数列名
            capacity_col: 容量列名
            figsize: 图形大小
            
        Returns:
            matplotlib.Figure: 图形对象
        """
        try:
            if cycle_col not in self.data.columns or capacity_col not in self.data.columns:
                raise ValueError(f"列 {cycle_col} 或 {capacity_col} 不存在")
            
            # 计算SOH
            initial_capacity = self.data[self.data[cycle_col] == self.data[cycle_col].min()][capacity_col].mean()
            soh_data = self.data.copy()
            soh_data['SOH'] = soh_data[capacity_col] / initial_capacity
            
            # 按循环次数分组，计算平均SOH
            soh_by_cycle = soh_data.groupby(cycle_col)['SOH'].mean().reset_index()
            
            fig, ax = plt.subplots(figsize=figsize)
            
            # 绘制SOH曲线
            ax.plot(soh_by_cycle[cycle_col], soh_by_cycle['SOH'], 
                   'g-', linewidth=2, marker='s', markersize=4, alpha=0.8)
            
            # 添加EOL阈值线
            ax.axhline(y=0.8, color='r', linestyle='--', alpha=0.8, label='EOL阈值 (80%)')
            
            ax.set_xlabel('循环次数')
            ax.set_ylabel('SOH')
            ax.set_title('电池健康状态(SOH)曲线')
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            
            logger.info("SOH曲线绘制完成")
            return fig
            
        except Exception as e:
            logger.error(f"绘制SOH曲线失败: {str(e)}")
            raise
    
    def plot_voltage_current_relationship(self, voltage_col: str, current_col: str, 
                                        cycle_col: str, figsize: tuple = (12, 8)) -> plt.Figure:
        """
        绘制电压-电流关系图
        
        Args:
            voltage_col: 电压列名
            current_col: 电流列名
            cycle_col: 循环次数列名
            figsize: 图形大小
            
        Returns:
            matplotlib.Figure: 图形对象
        """
        try:
            required_cols = [voltage_col, current_col, cycle_col]
            missing_cols = [col for col in required_cols if col not in self.data.columns]
            if missing_cols:
                raise ValueError(f"缺少列: {missing_cols}")
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
            
            # 1. 电压-电流散点图
            scatter = ax1.scatter(self.data[voltage_col], self.data[current_col], 
                                c=self.data[cycle_col], cmap='viridis', alpha=0.6, s=20)
            ax1.set_xlabel('电压')
            ax1.set_ylabel('电流')
            ax1.set_title('电压-电流关系 (颜色表示循环次数)')
            plt.colorbar(scatter, ax=ax1, label='循环次数')
            
            # 2. 电压随循环次数变化
            voltage_by_cycle = self.data.groupby(cycle_col)[voltage_col].mean().reset_index()
            ax2.plot(voltage_by_cycle[cycle_col], voltage_by_cycle[voltage_col], 'b-', linewidth=2)
            ax2.set_xlabel('循环次数')
            ax2.set_ylabel('平均电压')
            ax2.set_title('电压随循环次数变化')
            ax2.grid(True, alpha=0.3)
            
            # 3. 电流随循环次数变化
            current_by_cycle = self.data.groupby(cycle_col)[current_col].mean().reset_index()
            ax3.plot(current_by_cycle[cycle_col], current_by_cycle[current_col], 'r-', linewidth=2)
            ax3.set_xlabel('循环次数')
            ax3.set_ylabel('平均电流')
            ax3.set_title('电流随循环次数变化')
            ax3.grid(True, alpha=0.3)
            
            # 4. 电压和电流的分布对比
            ax4.hist(self.data[voltage_col], bins=50, alpha=0.7, label='电压', density=True)
            ax4_twin = ax4.twinx()
            ax4_twin.hist(self.data[current_col], bins=50, alpha=0.7, color='red', label='电流', density=True)
            ax4.set_xlabel('数值')
            ax4.set_ylabel('电压密度', color='blue')
            ax4_twin.set_ylabel('电流密度', color='red')
            ax4.set_title('电压和电流分布对比')
            
            plt.tight_layout()
            
            logger.info("电压-电流关系图绘制完成")
            return fig
            
        except Exception as e:
            logger.error(f"绘制电压-电流关系图失败: {str(e)}")
            raise
    
    def plot_time_series_analysis(self, time_col: str, value_cols: List[str], 
                                 figsize: tuple = (15, 10)) -> plt.Figure:
        """
        绘制时间序列分析图
        
        Args:
            time_col: 时间列名
            value_cols: 要分析的数值列名列表
            figsize: 图形大小
            
        Returns:
            matplotlib.Figure: 图形对象
        """
        try:
            if time_col not in self.data.columns:
                raise ValueError(f"时间列 {time_col} 不存在")
            
            missing_cols = [col for col in value_cols if col not in self.data.columns]
            if missing_cols:
                raise ValueError(f"缺少列: {missing_cols}")
            
            n_plots = len(value_cols)
            fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
            
            if n_plots == 1:
                axes = [axes]
            
            for i, col in enumerate(value_cols):
                axes[i].plot(self.data[time_col], self.data[col], linewidth=1, alpha=0.8)
                axes[i].set_ylabel(col)
                axes[i].set_title(f'{col} 时间序列')
                axes[i].grid(True, alpha=0.3)
                
                # 添加移动平均线
                if len(self.data) > 100:
                    window = min(100, len(self.data) // 10)
                    rolling_mean = self.data[col].rolling(window=window).mean()
                    axes[i].plot(self.data[time_col], rolling_mean, 'r-', linewidth=2, 
                               alpha=0.8, label=f'移动平均 (窗口={window})')
                    axes[i].legend()
            
            axes[-1].set_xlabel('时间')
            plt.tight_layout()
            
            logger.info(f"时间序列分析图绘制完成，包含 {n_plots} 个变量")
            return fig
            
        except Exception as e:
            logger.error(f"绘制时间序列分析图失败: {str(e)}")
            raise
    
    def generate_summary_statistics(self) -> pd.DataFrame:
        """
        生成汇总统计信息
        
        Returns:
            pd.DataFrame: 统计摘要
        """
        try:
            numeric_data = self.data.select_dtypes(include=[np.number])
            
            if numeric_data.empty:
                raise ValueError("数据中没有数值列")
            
            summary = numeric_data.describe()
            
            # 添加额外的统计信息
            additional_stats = pd.DataFrame({
                'skewness': numeric_data.skew(),
                'kurtosis': numeric_data.kurtosis(),
                'missing_count': numeric_data.isnull().sum(),
                'missing_percentage': (numeric_data.isnull().sum() / len(numeric_data)) * 100
            }).T
            
            summary = pd.concat([summary, additional_stats])
            
            logger.info("汇总统计信息生成完成")
            return summary
            
        except Exception as e:
            logger.error(f"生成汇总统计信息失败: {str(e)}")
            raise
    
    def detect_anomalies(self, columns: List[str], method: str = 'zscore', 
                        threshold: float = 3.0) -> pd.DataFrame:
        """
        检测异常值
        
        Args:
            columns: 要检测的列名列表
            method: 检测方法 ('zscore' 或 'iqr')
            threshold: 阈值
            
        Returns:
            pd.DataFrame: 异常值检测结果
        """
        try:
            anomalies = pd.DataFrame()
            
            for col in columns:
                if col not in self.data.columns:
                    continue
                
                if method == 'zscore':
                    from scipy import stats
                    z_scores = np.abs(stats.zscore(self.data[col].dropna()))
                    anomaly_mask = z_scores > threshold
                    
                elif method == 'iqr':
                    Q1 = self.data[col].quantile(0.25)
                    Q3 = self.data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    anomaly_mask = (self.data[col] < lower_bound) | (self.data[col] > upper_bound)
                
                else:
                    raise ValueError(f"不支持的检测方法: {method}")
                
                # 记录异常值信息
                anomaly_indices = self.data[anomaly_mask].index
                for idx in anomaly_indices:
                    anomalies = pd.concat([anomalies, pd.DataFrame({
                        'index': [idx],
                        'column': [col],
                        'value': [self.data.loc[idx, col]],
                        'method': [method],
                        'threshold': [threshold]
                    })], ignore_index=True)
            
            logger.info(f"异常值检测完成，发现 {len(anomalies)} 个异常值")
            return anomalies
            
        except Exception as e:
            logger.error(f"异常值检测失败: {str(e)}")
            raise

