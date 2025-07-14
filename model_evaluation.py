# coding: utf-8
"""
模型评估模块
提供模型性能评估、优化和比较功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
import warnings
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, learning_curve
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model):
        """
        初始化模型评估器
        
        Args:
            model: 要评估的模型对象
        """
        self.model = model
        self.optimization_results = {}
        
    def cross_validate(self, cv: int = 5, scoring: str = 'r2') -> Dict[str, float]:
        """
        交叉验证评估
        
        Args:
            cv: 交叉验证折数
            scoring: 评分指标
            
        Returns:
            Dict[str, float]: 交叉验证结果
        """
        try:
            if self.model.model is None:
                raise ValueError("模型未训练")
            
            # 准备数据
            X = np.vstack([self.model.X_train, self.model.X_test])
            y = np.hstack([self.model.y_train, self.model.y_test])
            
            # 执行交叉验证
            cv_scores = cross_val_score(self.model.model, X, y, cv=cv, scoring=scoring)
            
            results = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores.tolist()
            }
            
            logger.info(f"交叉验证完成: {scoring} = {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"交叉验证失败: {str(e)}")
            raise
    
    def optimize_hyperparameters(self, search_method: str = 'grid', 
                                n_iter: int = 50, cv: int = 5, 
                                scoring: str = 'r2') -> Any:
        """
        超参数优化
        
        Args:
            search_method: 搜索方法 ('grid', 'random', 'bayesian')
            n_iter: 随机搜索迭代次数
            cv: 交叉验证折数
            scoring: 评分指标
            
        Returns:
            优化后的模型
        """
        try:
            logger.info(f"开始超参数优化，使用 {search_method} 搜索...")
            
            # 准备数据
            X = np.vstack([self.model.X_train, self.model.X_test])
            y = np.hstack([self.model.y_train, self.model.y_test])
            
            # 定义参数网格
            param_grids = self._get_param_grids()
            
            if self.model.model_type not in param_grids:
                raise ValueError(f"不支持 {self.model.model_type} 的超参数优化")
            
            param_grid = param_grids[self.model.model_type]
            
            # 选择搜索方法
            if search_method == 'grid':
                search = GridSearchCV(
                    self.model.model, param_grid, cv=cv, scoring=scoring, n_jobs=-1
                )
            elif search_method == 'random':
                search = RandomizedSearchCV(
                    self.model.model, param_grid, n_iter=n_iter, cv=cv, 
                    scoring=scoring, n_jobs=-1, random_state=42
                )
            elif search_method == 'bayesian':
                # 这里可以集成贝叶斯优化库，如scikit-optimize
                logger.warning("贝叶斯优化暂未实现，使用随机搜索代替")
                search = RandomizedSearchCV(
                    self.model.model, param_grid, n_iter=n_iter, cv=cv, 
                    scoring=scoring, n_jobs=-1, random_state=42
                )
            else:
                raise ValueError(f"不支持的搜索方法: {search_method}")
            
            # 执行搜索
            search.fit(X, y)
            
            # 保存优化结果
            self.optimization_results['hyperparameter_optimization'] = {
                'best_params': search.best_params_,
                'best_score': search.best_score_,
                'search_method': search_method
            }
            
            # 创建优化后的模型
            optimized_model = self.model.__class__()
            optimized_model.model = search.best_estimator_
            optimized_model.model_type = self.model.model_type
            optimized_model.scaler = self.model.scaler
            optimized_model.feature_cols = self.model.feature_cols
            optimized_model.target_col = self.model.target_col
            optimized_model.X_train = self.model.X_train
            optimized_model.X_test = self.model.X_test
            optimized_model.y_train = self.model.y_train
            optimized_model.y_test = self.model.y_test
            
            # 重新预测
            optimized_model.y_pred = optimized_model.predict(optimized_model.X_test)
            
            logger.info(f"超参数优化完成，最佳分数: {search.best_score_:.4f}")
            logger.info(f"最佳参数: {search.best_params_}")
            
            return optimized_model
            
        except Exception as e:
            logger.error(f"超参数优化失败: {str(e)}")
            raise
    
    def select_features(self, method: str = 'rfe', n_features: int = 10) -> Any:
        """
        特征选择
        
        Args:
            method: 选择方法 ('rfe', 'univariate', 'importance')
            n_features: 选择的特征数量
            
        Returns:
            特征选择后的模型
        """
        try:
            logger.info(f"开始特征选择，使用 {method} 方法...")
            
            # 准备数据
            X = np.vstack([self.model.X_train, self.model.X_test])
            y = np.hstack([self.model.y_train, self.model.y_test])
            
            if method == 'rfe':
                # 递归特征消除
                selector = RFE(self.model.model, n_features_to_select=n_features)
                
            elif method == 'univariate':
                # 单变量特征选择
                selector = SelectKBest(score_func=f_regression, k=n_features)
                
            elif method == 'importance':
                # 基于特征重要性选择
                if not hasattr(self.model.model, 'feature_importances_'):
                    raise ValueError("模型不支持特征重要性")
                
                # 获取特征重要性
                importances = self.model.model.feature_importances_
                indices = np.argsort(importances)[::-1][:n_features]
                
                # 创建选择器
                class ImportanceSelector:
                    def __init__(self, indices):
                        self.indices = indices
                    
                    def fit_transform(self, X, y=None):
                        return X[:, self.indices]
                    
                    def transform(self, X):
                        return X[:, self.indices]
                    
                    def get_support(self):
                        support = np.zeros(X.shape[1], dtype=bool)
                        support[self.indices] = True
                        return support
                
                selector = ImportanceSelector(indices)
                
            else:
                raise ValueError(f"不支持的特征选择方法: {method}")
            
            # 执行特征选择
            if method != 'importance':
                X_selected = selector.fit_transform(X, y)
                selected_features = selector.get_support()
            else:
                X_selected = selector.fit_transform(X, y)
                selected_features = selector.get_support()
            
            # 获取选择的特征名称
            selected_feature_names = [self.model.feature_cols[i] for i in range(len(selected_features)) if selected_features[i]]
            
            # 保存特征选择结果
            self.optimization_results['feature_selection'] = {
                'method': method,
                'n_features': n_features,
                'selected_features': selected_feature_names
            }
            
            # 创建新模型
            from sklearn.model_selection import train_test_split
            
            X_train_selected, X_test_selected, y_train, y_test = train_test_split(
                X_selected, y, test_size=0.2, random_state=42
            )
            
            # 重新训练模型
            new_model = self.model.__class__()
            new_model.model_type = self.model.model_type
            new_model.feature_cols = selected_feature_names
            new_model.target_col = self.model.target_col
            new_model.X_train = X_train_selected
            new_model.X_test = X_test_selected
            new_model.y_train = y_train
            new_model.y_test = y_test
            
            # 重新训练
            if self.model.model_type == 'SVR':
                from sklearn.svm import SVR
                new_model.model = SVR()
            elif self.model.model_type == 'RandomForest':
                from sklearn.ensemble import RandomForestRegressor
                new_model.model = RandomForestRegressor(random_state=42)
            elif self.model.model_type == 'XGBoost':
                import xgboost as xgb
                new_model.model = xgb.XGBRegressor(random_state=42)
            elif self.model.model_type == 'LightGBM':
                import lightgbm as lgb
                new_model.model = lgb.LGBMRegressor(random_state=42)
            
            new_model.model.fit(X_train_selected, y_train)
            new_model.y_pred = new_model.predict(X_test_selected)
            
            logger.info(f"特征选择完成，选择了 {len(selected_feature_names)} 个特征")
            logger.info(f"选择的特征: {selected_feature_names}")
            
            return new_model
            
        except Exception as e:
            logger.error(f"特征选择失败: {str(e)}")
            raise
    
    def build_ensemble(self, method: str = 'voting', base_models: List[str] = None) -> Any:
        """
        构建集成模型
        
        Args:
            method: 集成方法 ('voting', 'stacking', 'bagging')
            base_models: 基础模型列表
            
        Returns:
            集成模型
        """
        try:
            logger.info(f"开始构建集成模型，使用 {method} 方法...")
            
            if base_models is None:
                base_models = ['SVR', 'RandomForest', 'XGBoost']
            
            # 准备数据
            X_train = self.model.X_train
            X_test = self.model.X_test
            y_train = self.model.y_train
            y_test = self.model.y_test
            
            # 创建基础模型
            estimators = []
            
            for model_name in base_models:
                if model_name == 'SVR':
                    from sklearn.svm import SVR
                    estimator = ('svr', SVR())
                elif model_name == 'RandomForest':
                    from sklearn.ensemble import RandomForestRegressor
                    estimator = ('rf', RandomForestRegressor(random_state=42))
                elif model_name == 'XGBoost':
                    import xgboost as xgb
                    estimator = ('xgb', xgb.XGBRegressor(random_state=42))
                elif model_name == 'LightGBM':
                    import lightgbm as lgb
                    estimator = ('lgb', lgb.LGBMRegressor(random_state=42))
                else:
                    logger.warning(f"不支持的模型: {model_name}")
                    continue
                
                estimators.append(estimator)
            
            # 构建集成模型
            if method == 'voting':
                ensemble_model = VotingRegressor(estimators=estimators)
                
            elif method == 'stacking':
                from sklearn.ensemble import StackingRegressor
                from sklearn.linear_model import LinearRegression
                ensemble_model = StackingRegressor(
                    estimators=estimators,
                    final_estimator=LinearRegression()
                )
                
            elif method == 'bagging':
                from sklearn.ensemble import BaggingRegressor
                # 使用第一个模型作为基础估计器
                base_estimator = estimators[0][1]
                ensemble_model = BaggingRegressor(
                    base_estimator=base_estimator,
                    n_estimators=len(estimators),
                    random_state=42
                )
                
            else:
                raise ValueError(f"不支持的集成方法: {method}")
            
            # 训练集成模型
            ensemble_model.fit(X_train, y_train)
            
            # 创建新的模型对象
            new_model = self.model.__class__()
            new_model.model = ensemble_model
            new_model.model_type = f'Ensemble_{method}'
            new_model.scaler = self.model.scaler
            new_model.feature_cols = self.model.feature_cols
            new_model.target_col = self.model.target_col
            new_model.X_train = X_train
            new_model.X_test = X_test
            new_model.y_train = y_train
            new_model.y_test = y_test
            new_model.y_pred = ensemble_model.predict(X_test)
            
            # 保存集成结果
            self.optimization_results['ensemble'] = {
                'method': method,
                'base_models': base_models
            }
            
            logger.info(f"集成模型构建完成，使用了 {len(base_models)} 个基础模型")
            
            return new_model
            
        except Exception as e:
            logger.error(f"构建集成模型失败: {str(e)}")
            raise
    
    def plot_learning_curve(self, cv: int = 5, figsize: tuple = (10, 6)) -> plt.Figure:
        """
        绘制学习曲线
        
        Args:
            cv: 交叉验证折数
            figsize: 图形大小
            
        Returns:
            matplotlib.Figure: 图形对象
        """
        try:
            # 准备数据
            X = np.vstack([self.model.X_train, self.model.X_test])
            y = np.hstack([self.model.y_train, self.model.y_test])
            
            # 计算学习曲线
            train_sizes, train_scores, val_scores = learning_curve(
                self.model.model, X, y, cv=cv, n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 10),
                scoring='r2'
            )
            
            # 计算均值和标准差
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            # 绘制学习曲线
            fig, ax = plt.subplots(figsize=figsize)
            
            ax.plot(train_sizes, train_mean, 'o-', color='blue', label='训练分数')
            ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                           alpha=0.1, color='blue')
            
            ax.plot(train_sizes, val_mean, 'o-', color='red', label='验证分数')
            ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                           alpha=0.1, color='red')
            
            ax.set_xlabel('训练样本数')
            ax.set_ylabel('R² 分数')
            ax.set_title('学习曲线')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            logger.error(f"绘制学习曲线失败: {str(e)}")
            raise
    
    def compare_models(self, models: List[Any], model_names: List[str] = None,
                      figsize: tuple = (12, 8)) -> plt.Figure:
        """
        比较多个模型的性能
        
        Args:
            models: 模型列表
            model_names: 模型名称列表
            figsize: 图形大小
            
        Returns:
            matplotlib.Figure: 图形对象
        """
        try:
            if model_names is None:
                model_names = [f'Model_{i+1}' for i in range(len(models))]
            
            # 收集评估指标
            metrics_data = []
            
            for i, model in enumerate(models):
                if hasattr(model, 'y_pred') and model.y_pred is not None:
                    metrics = model.evaluate_model()
                    metrics['model'] = model_names[i]
                    metrics_data.append(metrics)
            
            if not metrics_data:
                raise ValueError("没有可比较的模型")
            
            # 创建DataFrame
            metrics_df = pd.DataFrame(metrics_data)
            
            # 绘制比较图
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
            
            # R² 比较
            ax1.bar(metrics_df['model'], metrics_df['r2'])
            ax1.set_title('R² 分数比较')
            ax1.set_ylabel('R²')
            ax1.tick_params(axis='x', rotation=45)
            
            # RMSE 比较
            ax2.bar(metrics_df['model'], metrics_df['rmse'])
            ax2.set_title('RMSE 比较')
            ax2.set_ylabel('RMSE')
            ax2.tick_params(axis='x', rotation=45)
            
            # MAE 比较
            ax3.bar(metrics_df['model'], metrics_df['mae'])
            ax3.set_title('MAE 比较')
            ax3.set_ylabel('MAE')
            ax3.tick_params(axis='x', rotation=45)
            
            # MAPE 比较
            ax4.bar(metrics_df['model'], metrics_df['mape'])
            ax4.set_title('MAPE 比较')
            ax4.set_ylabel('MAPE (%)')
            ax4.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            logger.info(f"模型比较完成，比较了 {len(models)} 个模型")
            
            return fig
            
        except Exception as e:
            logger.error(f"模型比较失败: {str(e)}")
            raise
    
    def _get_param_grids(self) -> Dict[str, Dict]:
        """获取各模型的参数网格"""
        return {
            'SVR': {
                'kernel': ['rbf', 'linear', 'poly'],
                'C': [0.1, 1, 10, 100],
                'epsilon': [0.01, 0.1, 0.2]
            },
            'RandomForest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 9],
                'subsample': [0.8, 0.9, 1.0]
            },
            'LightGBM': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 9],
                'num_leaves': [31, 50, 100]
            }
        }
    
    def generate_evaluation_report(self) -> Dict[str, Any]:
        """
        生成评估报告
        
        Returns:
            Dict[str, Any]: 评估报告
        """
        try:
            report = {
                'model_info': {
                    'model_type': self.model.model_type,
                    'feature_count': len(self.model.feature_cols),
                    'training_samples': len(self.model.y_train),
                    'test_samples': len(self.model.y_test)
                },
                'performance_metrics': self.model.evaluate_model(),
                'optimization_results': self.optimization_results
            }
            
            # 添加交叉验证结果
            try:
                cv_results = self.cross_validate()
                report['cross_validation'] = cv_results
            except:
                report['cross_validation'] = None
            
            logger.info("评估报告生成完成")
            return report
            
        except Exception as e:
            logger.error(f"生成评估报告失败: {str(e)}")
            raise

