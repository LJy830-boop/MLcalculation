# coding: utf-8
"""
服务器连接管理模块
提供SSH连接、文件传输、远程命令执行等功能
"""

import os
import json
import time
import threading
from typing import Dict, List, Optional, Tuple, Any
import paramiko
from scp import SCPClient
import streamlit as st
from cryptography.fernet import Fernet
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServerConnectionManager:
    """服务器连接管理器"""
    
    def __init__(self):
        self.ssh_client = None
        self.scp_client = None
        self.is_connected = False
        self.connection_info = {}
        self.last_activity = None
        
    def connect(self, host: str, port: int = 22, username: str = None, 
                password: str = None, key_file: str = None, timeout: int = 30) -> bool:
        """
        连接到SSH服务器
        
        Args:
            host: 服务器地址
            port: SSH端口
            username: 用户名
            password: 密码
            key_file: SSH密钥文件路径
            timeout: 连接超时时间
            
        Returns:
            bool: 连接是否成功
        """
        try:
            # 创建SSH客户端
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # 连接参数
            connect_kwargs = {
                'hostname': host,
                'port': port,
                'username': username,
                'timeout': timeout
            }
            
            # 选择认证方式
            if key_file and os.path.exists(key_file):
                connect_kwargs['key_filename'] = key_file
                logger.info(f"使用SSH密钥认证: {key_file}")
            elif password:
                connect_kwargs['password'] = password
                logger.info("使用密码认证")
            else:
                raise ValueError("必须提供密码或SSH密钥文件")
            
            # 建立连接
            self.ssh_client.connect(**connect_kwargs)
            
            # 创建SCP客户端
            self.scp_client = SCPClient(self.ssh_client.get_transport())
            
            # 更新连接状态
            self.is_connected = True
            self.connection_info = {
                'host': host,
                'port': port,
                'username': username,
                'connected_at': time.time()
            }
            self.last_activity = time.time()
            
            logger.info(f"成功连接到服务器: {username}@{host}:{port}")
            return True
            
        except Exception as e:
            logger.error(f"连接失败: {str(e)}")
            self.disconnect()
            return False
    
    def disconnect(self):
        """断开连接"""
        try:
            if self.scp_client:
                self.scp_client.close()
                self.scp_client = None
                
            if self.ssh_client:
                self.ssh_client.close()
                self.ssh_client = None
                
            self.is_connected = False
            self.connection_info = {}
            self.last_activity = None
            
            logger.info("已断开服务器连接")
            
        except Exception as e:
            logger.error(f"断开连接时出错: {str(e)}")
    
    def test_connection(self) -> Dict[str, Any]:
        """
        测试连接状态
        
        Returns:
            Dict: 连接测试结果
        """
        if not self.is_connected:
            return {
                'status': 'disconnected',
                'message': '未连接到服务器'
            }
        
        try:
            # 执行简单命令测试连接
            start_time = time.time()
            stdin, stdout, stderr = self.ssh_client.exec_command('echo "connection_test"')
            result = stdout.read().decode().strip()
            latency = (time.time() - start_time) * 1000
            
            if result == "connection_test":
                self.last_activity = time.time()
                return {
                    'status': 'connected',
                    'message': '连接正常',
                    'latency': f"{latency:.2f}ms",
                    'server_info': self.connection_info
                }
            else:
                return {
                    'status': 'error',
                    'message': '连接异常'
                }
                
        except Exception as e:
            logger.error(f"连接测试失败: {str(e)}")
            return {
                'status': 'error',
                'message': f'连接测试失败: {str(e)}'
            }
    
    def get_server_info(self) -> Dict[str, str]:
        """
        获取服务器信息
        
        Returns:
            Dict: 服务器信息
        """
        if not self.is_connected:
            return {}
        
        try:
            info = {}
            
            # 获取系统信息
            commands = {
                'hostname': 'hostname',
                'os': 'uname -s',
                'kernel': 'uname -r',
                'architecture': 'uname -m',
                'cpu_info': 'cat /proc/cpuinfo | grep "model name" | head -1 | cut -d: -f2',
                'memory': 'free -h | grep Mem | awk \'{print $2}\'',
                'disk': 'df -h / | tail -1 | awk \'{print $2}\'',
                'python_version': 'python3 --version 2>/dev/null || python --version'
            }
            
            for key, command in commands.items():
                try:
                    stdin, stdout, stderr = self.ssh_client.exec_command(command)
                    result = stdout.read().decode().strip()
                    if result:
                        info[key] = result
                except:
                    info[key] = 'N/A'
            
            return info
            
        except Exception as e:
            logger.error(f"获取服务器信息失败: {str(e)}")
            return {}


class FileTransferManager:
    """文件传输管理器"""
    
    def __init__(self, connection_manager: ServerConnectionManager):
        self.connection_manager = connection_manager
    
    def upload_file(self, local_path: str, remote_path: str, 
                   progress_callback=None) -> bool:
        """
        上传文件到服务器
        
        Args:
            local_path: 本地文件路径
            remote_path: 远程文件路径
            progress_callback: 进度回调函数
            
        Returns:
            bool: 上传是否成功
        """
        if not self.connection_manager.is_connected:
            logger.error("未连接到服务器")
            return False
        
        try:
            if not os.path.exists(local_path):
                logger.error(f"本地文件不存在: {local_path}")
                return False
            
            # 确保远程目录存在
            remote_dir = os.path.dirname(remote_path)
            if remote_dir:
                self._ensure_remote_directory(remote_dir)
            
            # 上传文件
            if progress_callback:
                self.connection_manager.scp_client.put(
                    local_path, remote_path, progress=progress_callback
                )
            else:
                self.connection_manager.scp_client.put(local_path, remote_path)
            
            logger.info(f"文件上传成功: {local_path} -> {remote_path}")
            return True
            
        except Exception as e:
            logger.error(f"文件上传失败: {str(e)}")
            return False
    
    def download_file(self, remote_path: str, local_path: str,
                     progress_callback=None) -> bool:
        """
        从服务器下载文件
        
        Args:
            remote_path: 远程文件路径
            local_path: 本地文件路径
            progress_callback: 进度回调函数
            
        Returns:
            bool: 下载是否成功
        """
        if not self.connection_manager.is_connected:
            logger.error("未连接到服务器")
            return False
        
        try:
            # 确保本地目录存在
            local_dir = os.path.dirname(local_path)
            if local_dir:
                os.makedirs(local_dir, exist_ok=True)
            
            # 下载文件
            if progress_callback:
                self.connection_manager.scp_client.get(
                    remote_path, local_path, progress=progress_callback
                )
            else:
                self.connection_manager.scp_client.get(remote_path, local_path)
            
            logger.info(f"文件下载成功: {remote_path} -> {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"文件下载失败: {str(e)}")
            return False
    
    def upload_directory(self, local_dir: str, remote_dir: str) -> bool:
        """
        上传目录到服务器
        
        Args:
            local_dir: 本地目录路径
            remote_dir: 远程目录路径
            
        Returns:
            bool: 上传是否成功
        """
        if not self.connection_manager.is_connected:
            logger.error("未连接到服务器")
            return False
        
        try:
            if not os.path.exists(local_dir):
                logger.error(f"本地目录不存在: {local_dir}")
                return False
            
            # 确保远程目录存在
            self._ensure_remote_directory(remote_dir)
            
            # 递归上传目录
            self.connection_manager.scp_client.put(
                local_dir, remote_dir, recursive=True
            )
            
            logger.info(f"目录上传成功: {local_dir} -> {remote_dir}")
            return True
            
        except Exception as e:
            logger.error(f"目录上传失败: {str(e)}")
            return False
    
    def _ensure_remote_directory(self, remote_dir: str):
        """确保远程目录存在"""
        try:
            stdin, stdout, stderr = self.connection_manager.ssh_client.exec_command(
                f'mkdir -p "{remote_dir}"'
            )
            stdout.read()  # 等待命令执行完成
        except Exception as e:
            logger.error(f"创建远程目录失败: {str(e)}")


class RemoteCommandExecutor:
    """远程命令执行器"""
    
    def __init__(self, connection_manager: ServerConnectionManager):
        self.connection_manager = connection_manager
        self.running_jobs = {}
    
    def execute_command(self, command: str, timeout: int = 300) -> Dict[str, Any]:
        """
        执行远程命令
        
        Args:
            command: 要执行的命令
            timeout: 超时时间（秒）
            
        Returns:
            Dict: 执行结果
        """
        if not self.connection_manager.is_connected:
            return {
                'success': False,
                'error': '未连接到服务器'
            }
        
        try:
            logger.info(f"执行远程命令: {command}")
            
            # 执行命令
            stdin, stdout, stderr = self.connection_manager.ssh_client.exec_command(
                command, timeout=timeout
            )
            
            # 获取输出
            stdout_data = stdout.read().decode('utf-8', errors='ignore')
            stderr_data = stderr.read().decode('utf-8', errors='ignore')
            exit_code = stdout.channel.recv_exit_status()
            
            result = {
                'success': exit_code == 0,
                'exit_code': exit_code,
                'stdout': stdout_data,
                'stderr': stderr_data,
                'command': command
            }
            
            if exit_code == 0:
                logger.info(f"命令执行成功: {command}")
            else:
                logger.error(f"命令执行失败: {command}, 退出码: {exit_code}")
            
            return result
            
        except Exception as e:
            logger.error(f"执行命令时出错: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'command': command
            }
    
    def execute_script(self, script_content: str, script_name: str = "temp_script.py",
                      timeout: int = 600) -> Dict[str, Any]:
        """
        执行Python脚本
        
        Args:
            script_content: 脚本内容
            script_name: 脚本文件名
            timeout: 超时时间（秒）
            
        Returns:
            Dict: 执行结果
        """
        if not self.connection_manager.is_connected:
            return {
                'success': False,
                'error': '未连接到服务器'
            }
        
        try:
            # 创建临时脚本文件
            remote_script_path = f"/tmp/{script_name}"
            
            # 上传脚本内容
            stdin, stdout, stderr = self.connection_manager.ssh_client.exec_command(
                f'cat > {remote_script_path}'
            )
            stdin.write(script_content)
            stdin.close()
            
            # 执行脚本
            command = f"cd /tmp && python3 {script_name}"
            result = self.execute_command(command, timeout)
            
            # 清理临时文件
            self.execute_command(f"rm -f {remote_script_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"执行脚本时出错: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def start_training_job(self, config: Dict[str, Any]) -> str:
        """
        启动训练任务
        
        Args:
            config: 训练配置
            
        Returns:
            str: 任务ID
        """
        job_id = f"job_{int(time.time())}"
        
        try:
            # 构建训练命令
            algorithm = config.get('algorithm', 'XGBoost')
            data_file = config.get('data_file', 'data.csv')
            output_dir = config.get('output_dir', '/tmp/training_output')
            
            # 创建训练脚本
            training_script = self._generate_training_script(config)
            
            # 执行训练任务（后台运行）
            command = f"nohup python3 -c \"{training_script}\" > /tmp/{job_id}.log 2>&1 & echo $!"
            result = self.execute_command(command)
            
            if result['success']:
                pid = result['stdout'].strip()
                self.running_jobs[job_id] = {
                    'pid': pid,
                    'config': config,
                    'start_time': time.time(),
                    'status': 'running'
                }
                logger.info(f"训练任务已启动: {job_id}, PID: {pid}")
                return job_id
            else:
                logger.error(f"启动训练任务失败: {result.get('error', 'Unknown error')}")
                return None
                
        except Exception as e:
            logger.error(f"启动训练任务时出错: {str(e)}")
            return None
    
    def monitor_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        监控任务状态
        
        Args:
            job_id: 任务ID
            
        Returns:
            Dict: 任务状态信息
        """
        if job_id not in self.running_jobs:
            return {
                'status': 'not_found',
                'message': '任务不存在'
            }
        
        job_info = self.running_jobs[job_id]
        pid = job_info['pid']
        
        try:
            # 检查进程是否还在运行
            result = self.execute_command(f"ps -p {pid}")
            
            if result['success'] and pid in result['stdout']:
                status = 'running'
                message = '任务正在运行'
            else:
                status = 'completed'
                message = '任务已完成'
                self.running_jobs[job_id]['status'] = 'completed'
            
            # 获取日志
            log_result = self.execute_command(f"tail -20 /tmp/{job_id}.log")
            log_content = log_result.get('stdout', '') if log_result['success'] else ''
            
            return {
                'status': status,
                'message': message,
                'job_id': job_id,
                'pid': pid,
                'start_time': job_info['start_time'],
                'running_time': time.time() - job_info['start_time'],
                'log': log_content
            }
            
        except Exception as e:
            logger.error(f"监控任务状态时出错: {str(e)}")
            return {
                'status': 'error',
                'message': f'监控失败: {str(e)}'
            }
    
    def _generate_training_script(self, config: Dict[str, Any]) -> str:
        """生成训练脚本"""
        algorithm = config.get('algorithm', 'XGBoost')
        
        script_template = f"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json

# 加载数据
data = pd.read_csv('{config.get("data_file", "data.csv")}')

# 准备特征和目标
feature_cols = {config.get('feature_cols', [])}
target_col = '{config.get('target_col', 'SOH')}'

X = data[feature_cols]
y = data[target_col]

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 训练模型
if '{algorithm}' == 'XGBoost':
    import xgboost as xgb
    model = xgb.XGBRegressor(**{config.get('model_params', {})})
elif '{algorithm}' == 'RandomForest':
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(**{config.get('model_params', {})})
elif '{algorithm}' == 'SVR':
    from sklearn.svm import SVR
    model = SVR(**{config.get('model_params', {})})

model.fit(X_train, y_train)

# 预测和评估
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 保存结果
results = {{
    'algorithm': '{algorithm}',
    'mse': float(mse),
    'r2': float(r2),
    'model_params': {config.get('model_params', {})}
}}

with open('/tmp/training_results.json', 'w') as f:
    json.dump(results, f)

# 保存模型
joblib.dump(model, '/tmp/trained_model.pkl')

print(f'Training completed. MSE: {{mse:.4f}}, R2: {{r2:.4f}}')
"""
        return script_template


class ServerConfigManager:
    """服务器配置管理器"""
    
    def __init__(self, config_file: str = "server_config.json"):
        self.config_file = config_file
        self.encryption_key = self._get_or_create_key()
        self.cipher = Fernet(self.encryption_key)
    
    def _get_or_create_key(self) -> bytes:
        """获取或创建加密密钥"""
        key_file = ".server_key"
        
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            return key
    
    def save_config(self, config: Dict[str, Any]) -> bool:
        """
        保存服务器配置
        
        Args:
            config: 配置字典
            
        Returns:
            bool: 保存是否成功
        """
        try:
            # 加密敏感信息
            encrypted_config = self._encrypt_sensitive_data(config)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(encrypted_config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"配置已保存到: {self.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"保存配置失败: {str(e)}")
            return False
    
    def load_config(self) -> Dict[str, Any]:
        """
        加载服务器配置
        
        Returns:
            Dict: 配置字典
        """
        try:
            if not os.path.exists(self.config_file):
                return self._get_default_config()
            
            with open(self.config_file, 'r', encoding='utf-8') as f:
                encrypted_config = json.load(f)
            
            # 解密敏感信息
            config = self._decrypt_sensitive_data(encrypted_config)
            
            logger.info(f"配置已从 {self.config_file} 加载")
            return config
            
        except Exception as e:
            logger.error(f"加载配置失败: {str(e)}")
            return self._get_default_config()
    
    def _encrypt_sensitive_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """加密敏感数据"""
        encrypted_config = config.copy()
        
        if 'servers' in encrypted_config:
            for server in encrypted_config['servers']:
                if 'password' in server and server['password']:
                    server['password'] = self.cipher.encrypt(
                        server['password'].encode()
                    ).decode()
                    server['_encrypted'] = True
        
        return encrypted_config
    
    def _decrypt_sensitive_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """解密敏感数据"""
        decrypted_config = config.copy()
        
        if 'servers' in decrypted_config:
            for server in decrypted_config['servers']:
                if server.get('_encrypted') and 'password' in server:
                    try:
                        server['password'] = self.cipher.decrypt(
                            server['password'].encode()
                        ).decode()
                        del server['_encrypted']
                    except:
                        server['password'] = ''
        
        return decrypted_config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "servers": [],
            "default_server": None,
            "training_config": {
                "algorithms": ["SVR", "RandomForest", "XGBoost", "LightGBM"],
                "default_algorithm": "XGBoost",
                "remote_work_dir": "/tmp/battery_training",
                "max_timeout": 3600
            }
        }
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, str]:
        """
        验证配置
        
        Args:
            config: 配置字典
            
        Returns:
            Tuple[bool, str]: (是否有效, 错误信息)
        """
        try:
            # 检查必需字段
            if 'servers' not in config:
                return False, "缺少servers配置"
            
            for i, server in enumerate(config['servers']):
                required_fields = ['name', 'host', 'username']
                for field in required_fields:
                    if field not in server:
                        return False, f"服务器{i+1}缺少{field}字段"
                
                # 检查认证方式
                if not server.get('password') and not server.get('key_file'):
                    return False, f"服务器{i+1}必须提供密码或SSH密钥文件"
            
            return True, "配置有效"
            
        except Exception as e:
            return False, f"配置验证失败: {str(e)}"

