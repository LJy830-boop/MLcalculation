# coding: utf-8
"""
电池特征提取模块
提供时域、频域、小波和增量特征提取功能
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import warnings
from scipy import signal, stats
from scipy.fft import fft, fftfreq
import pywt
from sklearn.preprocessing import StandardScaler
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BatteryFeatureExtractor:
    """电池特征提取器"""
    
    def __init__(self, data: pd.DataFrame):
        """
        初始化特征提取器
        
        Args:
            data: 电池数据
        """
        self.data = data.copy()
        self.features = pd.DataFrame()
        
    def extract_time_domain_features(self, cycle_col: str, voltage_col: str, 
                                   current_col: str, time_col: str, 
                                   capacity_col: str = None) -> pd.DataFrame:
        """
        提取时域特征
        
        Args:
            cycle_col: 循环次数列名
            voltage_col: 电压列名
            current_col: 电流列名
            time_col: 时间列名
            capacity_col: 容量列名（可选）
            
        Returns:
            pd.DataFrame: 时域特征
        """
        try:
            logger.info("开始提取时域特征...")
            
            time_features = []
            
            for cycle in self.data[cycle_col].unique():
                cycle_data = self.data[self.data[cycle_col] == cycle]
                
                if len(cycle_data) < 2:
                    continue
                
                features = {'cycle': cycle}
                
                # 电压特征
                voltage_values = cycle_data[voltage_col].values
                features.update({
                    'voltage_mean': np.mean(voltage_values),
                    'voltage_std': np.std(voltage_values),
                    'voltage_min': np.min(voltage_values),
                    'voltage_max': np.max(voltage_values),
                    'voltage_range': np.max(voltage_values) - np.min(voltage_values),
                    'voltage_skewness': stats.skew(voltage_values),
                    'voltage_kurtosis': stats.kurtosis(voltage_values),
                    'voltage_rms': np.sqrt(np.mean(voltage_values**2)),
                    'voltage_peak_to_peak': np.ptp(voltage_values)
                })
                
                # 电流特征
                current_values = cycle_data[current_col].values
                features.update({
                    'current_mean': np.mean(current_values),
                    'current_std': np.std(current_values),
                    'current_min': np.min(current_values),
                    'current_max': np.max(current_values),
                    'current_range': np.max(current_values) - np.min(current_values),
                    'current_skewness': stats.skew(current_values),
                    'current_kurtosis': stats.kurtosis(current_values),
                    'current_rms': np.sqrt(np.mean(current_values**2)),
                    'current_peak_to_peak': np.ptp(current_values)
                })
                
                # 功率特征
                power_values = voltage_values * current_values
                features.update({
                    'power_mean': np.mean(power_values),
                    'power_std': np.std(power_values),
                    'power_min': np.min(power_values),
                    'power_max': np.max(power_values),
                    'power_range': np.max(power_values) - np.min(power_values)
                })
                
                # 时间特征
                if time_col in cycle_data.columns:
                    time_values = cycle_data[time_col].values
                    if len(time_values) > 1:
                        if pd.api.types.is_datetime64_any_dtype(time_values):
                            # 时间戳类型
                            time_diff = np.diff(time_values).astype('timedelta64[s]').astype(float)
                        else:
                            # 数值类型
                            time_diff = np.diff(time_values)
                        
                        features.update({
                            'cycle_duration': np.sum(time_diff),
                            'avg_time_step': np.mean(time_diff),
                            'time_step_std': np.std(time_diff)
                        })
                
                # 容量特征
                if capacity_col and capacity_col in cycle_data.columns:
                    capacity_values = cycle_data[capacity_col].values
                    features.update({
                        'capacity_mean': np.mean(capacity_values),
                        'capacity_std': np.std(capacity_values),
                        'capacity_min': np.min(capacity_values),
                        'capacity_max': np.max(capacity_values),
                        'capacity_fade': np.max(capacity_values) - np.min(capacity_values)
                    })
                
                # 变化率特征
                if len(voltage_values) > 1:
                    voltage_diff = np.diff(voltage_values)
                    current_diff = np.diff(current_values)
                    
                    features.update({
                        'voltage_change_rate_mean': np.mean(voltage_diff),
                        'voltage_change_rate_std': np.std(voltage_diff),
                        'current_change_rate_mean': np.mean(current_diff),
                        'current_change_rate_std': np.std(current_diff)
                    })
                
                time_features.append(features)
            
            time_features_df = pd.DataFrame(time_features)
            
            # 合并到主特征DataFrame
            if self.features.empty:
                self.features = time_features_df
            else:
                self.features = pd.merge(self.features, time_features_df, on='cycle', how='outer')
            
            logger.info(f"时域特征提取完成，提取了 {len(time_features_df.columns)-1} 个特征")
            return time_features_df
            
        except Exception as e:
            logger.error(f"时域特征提取失败: {str(e)}")
            raise
    
    def extract_frequency_domain_features(self, cycle_col: str, voltage_col: str, 
                                        current_col: str, time_col: str) -> pd.DataFrame:
        """
        提取频域特征
        
        Args:
            cycle_col: 循环次数列名
            voltage_col: 电压列名
            current_col: 电流列名
            time_col: 时间列名
            
        Returns:
            pd.DataFrame: 频域特征
        """
        try:
            logger.info("开始提取频域特征...")
            
            freq_features = []
            
            for cycle in self.data[cycle_col].unique():
                cycle_data = self.data[self.data[cycle_col] == cycle]
                
                if len(cycle_data) < 4:  # 需要足够的数据点进行FFT
                    continue
                
                features = {'cycle': cycle}
                
                # 电压频域特征
                voltage_values = cycle_data[voltage_col].values
                voltage_fft = np.abs(fft(voltage_values))
                voltage_freqs = fftfreq(len(voltage_values))
                
                # 只取正频率部分
                positive_freqs = voltage_freqs[:len(voltage_freqs)//2]
                positive_fft = voltage_fft[:len(voltage_fft)//2]
                
                if len(positive_fft) > 0:
                    features.update({
                        'voltage_fft_mean': np.mean(positive_fft),
                        'voltage_fft_std': np.std(positive_fft),
                        'voltage_fft_max': np.max(positive_fft),
                        'voltage_dominant_freq': positive_freqs[np.argmax(positive_fft)] if len(positive_fft) > 0 else 0,
                        'voltage_spectral_centroid': np.sum(positive_freqs * positive_fft) / np.sum(positive_fft) if np.sum(positive_fft) > 0 else 0
                    })
                
                # 电流频域特征
                current_values = cycle_data[current_col].values
                current_fft = np.abs(fft(current_values))
                current_freqs = fftfreq(len(current_values))
                
                positive_freqs_current = current_freqs[:len(current_freqs)//2]
                positive_fft_current = current_fft[:len(current_fft)//2]
                
                if len(positive_fft_current) > 0:
                    features.update({
                        'current_fft_mean': np.mean(positive_fft_current),
                        'current_fft_std': np.std(positive_fft_current),
                        'current_fft_max': np.max(positive_fft_current),
                        'current_dominant_freq': positive_freqs_current[np.argmax(positive_fft_current)] if len(positive_fft_current) > 0 else 0,
                        'current_spectral_centroid': np.sum(positive_freqs_current * positive_fft_current) / np.sum(positive_fft_current) if np.sum(positive_fft_current) > 0 else 0
                    })
                
                # 功率谱密度特征
                if len(voltage_values) > 1:
                    freqs, psd = signal.periodogram(voltage_values)
                    if len(psd) > 0:
                        features.update({
                            'voltage_psd_mean': np.mean(psd),
                            'voltage_psd_max': np.max(psd),
                            'voltage_psd_peak_freq': freqs[np.argmax(psd)] if len(psd) > 0 else 0
                        })
                
                freq_features.append(features)
            
            freq_features_df = pd.DataFrame(freq_features)
            
            # 合并到主特征DataFrame
            if self.features.empty:
                self.features = freq_features_df
            else:
                self.features = pd.merge(self.features, freq_features_df, on='cycle', how='outer')
            
            logger.info(f"频域特征提取完成，提取了 {len(freq_features_df.columns)-1} 个特征")
            return freq_features_df
            
        except Exception as e:
            logger.error(f"频域特征提取失败: {str(e)}")
            raise
    
    def extract_wavelet_features(self, cycle_col: str, voltage_col: str, 
                               current_col: str, time_col: str, 
                               wavelet: str = 'db4') -> pd.DataFrame:
        """
        提取小波特征
        
        Args:
            cycle_col: 循环次数列名
            voltage_col: 电压列名
            current_col: 电流列名
            time_col: 时间列名
            wavelet: 小波类型
            
        Returns:
            pd.DataFrame: 小波特征
        """
        try:
            logger.info("开始提取小波特征...")
            
            wavelet_features = []
            
            for cycle in self.data[cycle_col].unique():
                cycle_data = self.data[self.data[cycle_col] == cycle]
                
                if len(cycle_data) < 4:
                    continue
                
                features = {'cycle': cycle}
                
                # 电压小波特征
                voltage_values = cycle_data[voltage_col].values
                try:
                    coeffs = pywt.wavedec(voltage_values, wavelet, level=3)
                    
                    for i, coeff in enumerate(coeffs):
                        features.update({
                            f'voltage_wavelet_level_{i}_mean': np.mean(np.abs(coeff)),
                            f'voltage_wavelet_level_{i}_std': np.std(coeff),
                            f'voltage_wavelet_level_{i}_energy': np.sum(coeff**2)
                        })
                except:
                    # 如果小波分解失败，使用默认值
                    for i in range(4):
                        features.update({
                            f'voltage_wavelet_level_{i}_mean': 0,
                            f'voltage_wavelet_level_{i}_std': 0,
                            f'voltage_wavelet_level_{i}_energy': 0
                        })
                
                # 电流小波特征
                current_values = cycle_data[current_col].values
                try:
                    coeffs = pywt.wavedec(current_values, wavelet, level=3)
                    
                    for i, coeff in enumerate(coeffs):
                        features.update({
                            f'current_wavelet_level_{i}_mean': np.mean(np.abs(coeff)),
                            f'current_wavelet_level_{i}_std': np.std(coeff),
                            f'current_wavelet_level_{i}_energy': np.sum(coeff**2)
                        })
                except:
                    # 如果小波分解失败，使用默认值
                    for i in range(4):
                        features.update({
                            f'current_wavelet_level_{i}_mean': 0,
                            f'current_wavelet_level_{i}_std': 0,
                            f'current_wavelet_level_{i}_energy': 0
                        })
                
                wavelet_features.append(features)
            
            wavelet_features_df = pd.DataFrame(wavelet_features)
            
            # 合并到主特征DataFrame
            if self.features.empty:
                self.features = wavelet_features_df
            else:
                self.features = pd.merge(self.features, wavelet_features_df, on='cycle', how='outer')
            
            logger.info(f"小波特征提取完成，提取了 {len(wavelet_features_df.columns)-1} 个特征")
            return wavelet_features_df
            
        except Exception as e:
            logger.error(f"小波特征提取失败: {str(e)}")
            raise
    
    def extract_ic_curve_features(self, cycle_col: str, voltage_col: str, 
                                 current_col: str, capacity_col: str) -> pd.DataFrame:
        """
        提取IC曲线特征
        
        Args:
            cycle_col: 循环次数列名
            voltage_col: 电压列名
            current_col: 电流列名
            capacity_col: 容量列名
            
        Returns:
            pd.DataFrame: IC曲线特征
        """
        try:
            logger.info("开始提取IC曲线特征...")
            
            ic_features = []
            
            for cycle in self.data[cycle_col].unique():
                cycle_data = self.data[self.data[cycle_col] == cycle]
                
                if len(cycle_data) < 3:
                    continue
                
                features = {'cycle': cycle}
                
                # 计算IC曲线 (dQ/dV)
                voltage_values = cycle_data[voltage_col].values
                capacity_values = cycle_data[capacity_col].values
                
                # 确保数据是单调的
                sorted_indices = np.argsort(voltage_values)
                voltage_sorted = voltage_values[sorted_indices]
                capacity_sorted = capacity_values[sorted_indices]
                
                # 计算dQ/dV
                if len(voltage_sorted) > 1:
                    dV = np.diff(voltage_sorted)
                    dQ = np.diff(capacity_sorted)
                    
                    # 避免除零
                    valid_indices = np.abs(dV) > 1e-6
                    if np.any(valid_indices):
                        ic_values = dQ[valid_indices] / dV[valid_indices]
                        
                        features.update({
                            'ic_mean': np.mean(ic_values),
                            'ic_std': np.std(ic_values),
                            'ic_min': np.min(ic_values),
                            'ic_max': np.max(ic_values),
                            'ic_range': np.max(ic_values) - np.min(ic_values),
                            'ic_skewness': stats.skew(ic_values),
                            'ic_kurtosis': stats.kurtosis(ic_values)
                        })
                        
                        # IC峰值特征
                        if len(ic_values) > 2:
                            peaks, _ = signal.find_peaks(ic_values, height=np.mean(ic_values))
                            features.update({
                                'ic_peak_count': len(peaks),
                                'ic_peak_height_mean': np.mean(ic_values[peaks]) if len(peaks) > 0 else 0,
                                'ic_peak_height_max': np.max(ic_values[peaks]) if len(peaks) > 0 else 0
                            })
                
                ic_features.append(features)
            
            ic_features_df = pd.DataFrame(ic_features)
            
            # 合并到主特征DataFrame
            if self.features.empty:
                self.features = ic_features_df
            else:
                self.features = pd.merge(self.features, ic_features_df, on='cycle', how='outer')
            
            logger.info(f"IC曲线特征提取完成，提取了 {len(ic_features_df.columns)-1} 个特征")
            return ic_features_df
            
        except Exception as e:
            logger.error(f"IC曲线特征提取失败: {str(e)}")
            raise
    
    def extract_incremental_features(self, cycle_col: str) -> pd.DataFrame:
        """
        提取增量特征（基于已有特征的变化率）
        
        Args:
            cycle_col: 循环次数列名
            
        Returns:
            pd.DataFrame: 增量特征
        """
        try:
            logger.info("开始提取增量特征...")
            
            if self.features.empty:
                logger.warning("没有基础特征，无法提取增量特征")
                return pd.DataFrame()
            
            # 按循环次数排序
            features_sorted = self.features.sort_values(cycle_col).copy()
            
            # 计算增量特征
            incremental_features = features_sorted.copy()
            
            # 对数值列计算差分
            numeric_cols = features_sorted.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col != cycle_col]
            
            for col in numeric_cols:
                # 一阶差分
                incremental_features[f'{col}_diff1'] = features_sorted[col].diff()
                
                # 二阶差分
                incremental_features[f'{col}_diff2'] = features_sorted[col].diff().diff()
                
                # 变化率
                incremental_features[f'{col}_pct_change'] = features_sorted[col].pct_change()
                
                # 累积变化
                incremental_features[f'{col}_cumsum'] = features_sorted[col].cumsum()
                
                # 滚动平均（窗口大小为3）
                if len(features_sorted) >= 3:
                    incremental_features[f'{col}_rolling_mean_3'] = features_sorted[col].rolling(window=3, min_periods=1).mean()
                    incremental_features[f'{col}_rolling_std_3'] = features_sorted[col].rolling(window=3, min_periods=1).std()
            
            # 填充NaN值
            incremental_features = incremental_features.fillna(0)
            
            self.features = incremental_features
            
            logger.info(f"增量特征提取完成，总特征数: {len(incremental_features.columns)-1}")
            return incremental_features
            
        except Exception as e:
            logger.error(f"增量特征提取失败: {str(e)}")
            raise
    
    def calculate_soh_features(self, cycle_col: str, capacity_col: str) -> pd.DataFrame:
        """
        计算SOH相关特征
        
        Args:
            cycle_col: 循环次数列名
            capacity_col: 容量列名
            
        Returns:
            pd.DataFrame: SOH特征
        """
        try:
            logger.info("开始计算SOH特征...")
            
            soh_features = []
            
            # 获取初始容量
            initial_capacity = self.data[self.data[cycle_col] == self.data[cycle_col].min()][capacity_col].mean()
            
            for cycle in self.data[cycle_col].unique():
                cycle_data = self.data[self.data[cycle_col] == cycle]
                
                features = {'cycle': cycle}
                
                # 当前容量
                current_capacity = cycle_data[capacity_col].mean()
                
                # SOH计算
                soh = current_capacity / initial_capacity
                features['SOH'] = soh
                
                # 容量退化特征
                features['capacity_fade'] = initial_capacity - current_capacity
                features['capacity_fade_rate'] = (initial_capacity - current_capacity) / cycle if cycle > 0 else 0
                
                # 相对于初始容量的百分比
                features['capacity_retention'] = soh * 100
                
                soh_features.append(features)
            
            soh_features_df = pd.DataFrame(soh_features)
            
            # 合并到主特征DataFrame
            if self.features.empty:
                self.features = soh_features_df
            else:
                self.features = pd.merge(self.features, soh_features_df, on='cycle', how='outer')
            
            logger.info("SOH特征计算完成")
            return soh_features_df
            
        except Exception as e:
            logger.error(f"SOH特征计算失败: {str(e)}")
            raise
    
    def select_features(self, target_col: str = 'SOH', method: str = 'correlation', 
                       n_features: int = 20) -> List[str]:
        """
        特征选择
        
        Args:
            target_col: 目标列名
            method: 选择方法 ('correlation', 'variance', 'mutual_info')
            n_features: 选择的特征数量
            
        Returns:
            List[str]: 选择的特征列名
        """
        try:
            if self.features.empty:
                logger.warning("没有特征数据")
                return []
            
            if target_col not in self.features.columns:
                logger.warning(f"目标列 {target_col} 不存在")
                return []
            
            # 获取数值特征列（排除目标列和循环列）
            feature_cols = [col for col in self.features.columns 
                          if col not in [target_col, 'cycle'] and 
                          self.features[col].dtype in [np.float64, np.int64]]
            
            if len(feature_cols) == 0:
                logger.warning("没有可用的特征列")
                return []
            
            if method == 'correlation':
                # 基于相关性选择
                correlations = self.features[feature_cols + [target_col]].corr()[target_col].abs()
                correlations = correlations.drop(target_col).sort_values(ascending=False)
                selected_features = correlations.head(n_features).index.tolist()
                
            elif method == 'variance':
                # 基于方差选择
                variances = self.features[feature_cols].var().sort_values(ascending=False)
                selected_features = variances.head(n_features).index.tolist()
                
            elif method == 'mutual_info':
                # 基于互信息选择
                from sklearn.feature_selection import mutual_info_regression
                
                X = self.features[feature_cols].fillna(0)
                y = self.features[target_col].fillna(0)
                
                mi_scores = mutual_info_regression(X, y)
                mi_df = pd.DataFrame({'feature': feature_cols, 'mi_score': mi_scores})
                mi_df = mi_df.sort_values('mi_score', ascending=False)
                selected_features = mi_df.head(n_features)['feature'].tolist()
                
            else:
                raise ValueError(f"不支持的特征选择方法: {method}")
            
            logger.info(f"特征选择完成，使用 {method} 方法选择了 {len(selected_features)} 个特征")
            return selected_features
            
        except Exception as e:
            logger.error(f"特征选择失败: {str(e)}")
            raise
    
    def get_feature_summary(self) -> Dict:
        """
        获取特征摘要
        
        Returns:
            Dict: 特征摘要信息
        """
        if self.features.empty:
            return {"message": "没有提取的特征"}
        
        summary = {
            "total_features": len(self.features.columns) - 1,  # 排除cycle列
            "total_cycles": len(self.features),
            "feature_types": {},
            "missing_values": self.features.isnull().sum().sum(),
            "feature_names": [col for col in self.features.columns if col != 'cycle']
        }
        
        # 按特征类型分类
        for col in self.features.columns:
            if col == 'cycle':
                continue
                
            if 'voltage' in col:
                summary["feature_types"]["voltage"] = summary["feature_types"].get("voltage", 0) + 1
            elif 'current' in col:
                summary["feature_types"]["current"] = summary["feature_types"].get("current", 0) + 1
            elif 'power' in col:
                summary["feature_types"]["power"] = summary["feature_types"].get("power", 0) + 1
            elif 'capacity' in col or 'SOH' in col:
                summary["feature_types"]["capacity"] = summary["feature_types"].get("capacity", 0) + 1
            elif 'fft' in col or 'freq' in col or 'spectral' in col:
                summary["feature_types"]["frequency"] = summary["feature_types"].get("frequency", 0) + 1
            elif 'wavelet' in col:
                summary["feature_types"]["wavelet"] = summary["feature_types"].get("wavelet", 0) + 1
            elif 'ic' in col:
                summary["feature_types"]["ic_curve"] = summary["feature_types"].get("ic_curve", 0) + 1
            elif 'diff' in col or 'pct_change' in col or 'rolling' in col:
                summary["feature_types"]["incremental"] = summary["feature_types"].get("incremental", 0) + 1
            else:
                summary["feature_types"]["other"] = summary["feature_types"].get("other", 0) + 1
        
        return summary
    
    def save_features(self, filepath: str):
        """保存特征到文件"""
        if not self.features.empty:
            self.features.to_csv(filepath, index=False)
            logger.info(f"特征已保存到: {filepath}")
        else:
            logger.warning("没有特征数据可保存")
    
    def load_features(self, filepath: str):
        """从文件加载特征"""
        try:
            self.features = pd.read_csv(filepath)
            logger.info(f"特征已从 {filepath} 加载")
        except Exception as e:
            logger.error(f"加载特征失败: {str(e)}")
            raise

