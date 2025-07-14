# 服务器连接架构设计

## 1. 架构概述

为电池寿命预测系统添加服务器连接功能，支持远程模型训练和数据处理。架构采用SSH协议进行安全连接，支持文件传输和远程命令执行。

## 2. 连接方式

### 2.1 SSH连接
- **协议**: SSH (Secure Shell)
- **端口**: 默认22，可配置
- **认证方式**: 
  - 用户名/密码认证
  - SSH密钥认证（推荐）
  - 双因子认证支持

### 2.2 文件传输
- **协议**: SFTP (SSH File Transfer Protocol)
- **功能**: 
  - 上传训练数据
  - 下载训练结果
  - 同步模型文件

### 2.3 远程命令执行
- **方式**: SSH命令执行
- **功能**:
  - 环境检查
  - 脚本执行
  - 训练任务启动
  - 状态监控

## 3. 系统组件

### 3.1 连接管理器 (ConnectionManager)
```python
class ServerConnectionManager:
    - connect(host, port, username, password/key)
    - disconnect()
    - test_connection()
    - get_connection_status()
```

### 3.2 文件传输器 (FileTransfer)
```python
class FileTransferManager:
    - upload_file(local_path, remote_path)
    - download_file(remote_path, local_path)
    - upload_directory(local_dir, remote_dir)
    - download_directory(remote_dir, local_dir)
```

### 3.3 远程执行器 (RemoteExecutor)
```python
class RemoteCommandExecutor:
    - execute_command(command)
    - execute_script(script_path)
    - start_training_job(config)
    - monitor_job_status(job_id)
```

### 3.4 配置管理器 (ConfigManager)
```python
class ServerConfigManager:
    - save_config(config)
    - load_config()
    - validate_config(config)
    - encrypt_credentials(credentials)
```

## 4. 用户界面设计

### 4.1 新增导航选项
在Streamlit侧边栏添加"服务器连接"选项，位于现有7个步骤之后。

### 4.2 服务器配置页面
- **连接参数配置**:
  - 服务器地址
  - 端口号
  - 用户名
  - 密码/SSH密钥
  - 连接超时设置

- **连接测试**:
  - 连接状态显示
  - 网络延迟测试
  - 服务器环境检查

### 4.3 远程训练页面
- **数据上传**:
  - 选择本地数据文件
  - 上传进度显示
  - 上传状态反馈

- **训练配置**:
  - 选择训练算法
  - 设置训练参数
  - 选择GPU/CPU资源

- **任务监控**:
  - 训练进度显示
  - 实时日志查看
  - 资源使用监控

## 5. 安全考虑

### 5.1 认证安全
- SSH密钥认证优于密码认证
- 支持密钥文件加密
- 连接超时自动断开

### 5.2 数据安全
- 传输过程加密
- 敏感信息本地加密存储
- 临时文件自动清理

### 5.3 访问控制
- 限制可执行命令范围
- 文件访问权限检查
- 操作日志记录

## 6. 错误处理

### 6.1 连接错误
- 网络不可达
- 认证失败
- 超时处理

### 6.2 传输错误
- 文件不存在
- 权限不足
- 磁盘空间不足

### 6.3 执行错误
- 命令不存在
- 脚本执行失败
- 资源不足

## 7. 配置文件格式

### 7.1 服务器配置 (server_config.json)
```json
{
  "servers": [
    {
      "name": "训练服务器1",
      "host": "192.168.1.100",
      "port": 22,
      "username": "user",
      "auth_type": "key",
      "key_file": "~/.ssh/id_rsa",
      "remote_work_dir": "/home/user/battery_training"
    }
  ],
  "default_server": "训练服务器1"
}
```

### 7.2 训练配置 (training_config.json)
```json
{
  "algorithms": ["SVR", "RandomForest", "XGBoost", "LSTM"],
  "default_algorithm": "XGBoost",
  "resource_limits": {
    "max_memory": "8GB",
    "max_cpu_cores": 4,
    "max_gpu_memory": "4GB"
  },
  "training_scripts": {
    "SVR": "train_svr.py",
    "RandomForest": "train_rf.py",
    "XGBoost": "train_xgb.py",
    "LSTM": "train_lstm.py"
  }
}
```

## 8. 部署考虑

### 8.1 依赖库
- paramiko (SSH连接)
- scp (文件传输)
- cryptography (加密)
- streamlit (UI)

### 8.2 环境要求
- Python 3.8+
- SSH客户端支持
- 网络连接

### 8.3 服务器要求
- SSH服务开启
- Python环境配置
- 必要的机器学习库安装

