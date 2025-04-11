# 导入必要的库
import torch  # PyTorch深度学习框架
from torch.optim import AdamW  # AdamW优化器
from transformers import get_linear_schedule_with_warmup  # 学习率调度器
from sklearn.metrics import accuracy_score, f1_score  # 评估指标
import numpy as np  # 数值计算库
from tqdm import tqdm  # 进度条显示
from model import MedicalIntentClassifier  # 自定义模型类
from data_processor import load_data, create_data_loaders  # 数据加载函数
from utils.logger import setup_logger  # 日志设置
import os  # 操作系统接口
from transformers import BertTokenizer  # 添加BertTokenizer
from utils.seed import set_seed  # 添加set_seed函数

# 定义早停类，用于防止过拟合
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):  # 增加耐心值和最小变化量
        self.patience = patience  # 容忍的验证集性能不提升的轮数
        self.min_delta = min_delta  # 认为性能提升的最小变化量
        self.counter = 0  # 计数器，记录性能未提升的轮数
        self.best_score = None  # 记录最佳验证集分数
        self.early_stop = False  # 是否触发早停的标志
        
    def __call__(self, val_score):
        if self.best_score is None:  # 第一次调用时
            self.best_score = val_score  # 直接记录分数
        elif val_score < self.best_score + self.min_delta:  # 如果当前分数没有显著提升
            self.counter += 1  # 增加计数器
            if self.counter >= self.patience:  # 如果连续patience轮都没有提升
                self.early_stop = True  # 触发早停
        else:  # 如果分数有提升
            self.best_score = val_score  # 更新最佳分数
            self.counter = 0  # 重置计数器

# 定义模型训练函数
def train_model(model, train_loader, val_loader, device, num_epochs=3, logger=None):
    torch.backends.cudnn.benchmark = True
    
    # 使用较小的学习率和较大的权重衰减
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.05)
    total_steps = len(train_loader) * num_epochs
    
    # 使用更长的预热期
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.2),
        num_training_steps=total_steps
    )
    
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)
    best_val_f1 = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        
        scaler = torch.cuda.amp.GradScaler()
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            current_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({'loss': current_loss})
            
            if batch_idx % 100 == 0 and logger:
                logger.info(f'Epoch {epoch + 1}, Batch {batch_idx}, Loss: {current_loss:.4f}')
        
        val_accuracy, val_f1 = evaluate_model(model, val_loader, device)
        epoch_msg = f'Epoch {epoch + 1} - Validation Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}'
        print(epoch_msg)
        if logger:
            logger.info(epoch_msg)
        
        early_stopping(val_f1)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model_save_path = 'best_model.pth'
            torch.save(model.state_dict(), model_save_path)
            if logger:
                logger.info(f'保存最佳模型，F1分数: {val_f1:.4f}')

# 定义模型评估函数
def evaluate_model(model, data_loader, device):
    model.eval()  # 设置为评估模式
    predictions = []  # 存储预测结果
    true_labels = []  # 存储真实标签
    
    with torch.no_grad():  # 禁用梯度计算
        for batch in data_loader:  # 遍历每个batch
            # 将数据移到指定设备
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast():  # 使用混合精度进行预测
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)  # 获取预测结果
            
            predictions.extend(preds.cpu().numpy())  # 收集预测结果
            true_labels.extend(labels.cpu().numpy())  # 收集真实标签
    
    # 计算评估指标
    accuracy = accuracy_score(true_labels, predictions)  # 计算准确率
    f1 = f1_score(true_labels, predictions, average='weighted')  # 计算F1分数
    
    return accuracy, f1

# 主函数
def main():
    # 设置随机种子
    set_seed(42)
    
    # 初始化日志记录器
    logger = setup_logger()
    logger.info("开始训练流程")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载数据
    logger.info("加载数据集...")
    train_data = load_data("data/IMCS-DAC_train.json")
    val_data = load_data("data/IMCS-DAC_dev.json")
    test_data = load_data("data/IMCS-DAC_test.json")
    
    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    # 创建数据加载器
    train_loader, val_loader, test_loader, num_labels = create_data_loaders(
        train_data, val_data, test_data, tokenizer, batch_size=32
    )
    
    # 初始化模型
    model = MedicalIntentClassifier(num_labels=num_labels)
    model.to(device)
    
    # 训练模型
    logger.info("开始训练...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=3,
        logger=logger
    )
    
    # 在测试集上评估
    logger.info("在测试集上评估...")
    test_accuracy, test_f1 = evaluate_model(model, test_loader, device)
    logger.info(f"测试集准确率: {test_accuracy:.4f}, F1分数: {test_f1:.4f}")
    
    # 保存模型
    logger.info("保存模型...")
    torch.save(model.state_dict(), "models/medical_intent_classifier.pth")
    logger.info("训练完成")

if __name__ == '__main__':
    main()  # 运行主函数