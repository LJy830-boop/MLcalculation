# coding: utf-8
"""
电池寿命预测系统安装配置
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="battery-prediction-system",
    version="1.0.0",
    author="浙江锋锂新能源科技有限公司-唐光盛团队",
    author_email="your-email@example.com",
    description="基于机器学习的电池寿命预测系统",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/battery-prediction-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "jupyter>=1.0.0",
        ],
        "gpu": [
            "tensorflow-gpu>=2.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "battery-prediction=streamlit_app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.toml", "*.txt", "*.md"],
    },
    zip_safe=False,
)

