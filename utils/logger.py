import logging
from pathlib import Path

def setup_logger(name='MedicalClassifier'):
    # 创建日志目录
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # 配置基础日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 主日志器配置
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 文件处理器（调试级别）
    file_handler = logging.FileHandler(log_dir / 'training.log')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # 控制台处理器（信息级别）
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # 避免重复添加处理器
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger