# -*- coding: utf-8 -*-
"""
医疗对话意图识别BERT微调完整实现
包含数据预处理、模型训练、预测全流程
"""

# 导入必要库
import os  # 新增os模块导入
import torch
import json  # 标准库：JSON文件处理
import re    # 标准库：正则表达式处理
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    BertTokenizer,
    BertForSequenceClassification
)
from torch.optim import AdamW
from typing import Optional, Dict  # 新增类型提示
import torch.nn as nn  # 新增这行导入语句

# 设置国内镜像源（新增环境变量配置）
# 原环境变量设置
# 正确修改后
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 将CONFIG字典定义移动到所有使用它的代码之前
CONFIG = {
    "pretrained": "bert-base-chinese",  # 修改为标准模型名称
    "max_len": 128,                     # 文本最大长度
    "batch_size": 16,                   # 训练批大小
    "epochs": 5,                        # 训练轮数
    "lr": 3e-5,                         # 基础学习率
    "class_names": [                    # 全部16个意图类别
        "Request-Symptom",
        "Inform-Symptom",
        "Request-Etiology",
        "Inform-Etiology",
        "Request-Basic_Information",
        "Inform-Basic_Information",
        "Request-Existing_Examination_and_Treatment",
        "Inform-Existing_Examination_and_Treatment",
        "Request-Drug_Recommendation",
        "Inform-Drug_Recommendation",
        "Request-Medical_Advice",
        "Inform-Medical_Advice",
        "Request-Precautions",
        "Inform-Precautions",
        "Diagnose",
        "Other"
    ],  
    "speaker_mapping": {"医生": 0, "患者": 1},  # 说话者编码映射
    "data_paths": {                      # 数据集路径配置
        "train": "/C:/Users/20583/Desktop/微调小队/weitiao_BERT--/src/datasets/IMCS-DAC_train.json",
        "dev": "/C:/Users/20583/Desktop/微调小队/weitiao_BERT--/src/datasets/IMCS-DAC_dev.json",
        "test": "/C:/Users/20583/Desktop/微调小队/weitiao_BERT--/src/datasets/IMCS-DAC_test.json",
        "output": "/C:/Users/20583/Desktop/微调小队/weitiao_BERT--/src/submission.json"      # 预测结果保存路径
    }
}
# 新增镜像源参数（在模型加载处）
tokenizer = BertTokenizer.from_pretrained(
    CONFIG["pretrained"],
    use_auth_token=False,
    mirror="https://hf-mirror.com"  # 新增镜像参数
)

# 初始化BERT组件（修改模型路径为镜像站地址）
# 配置文件（修改预训练模型路径）

# 初始化BERT组件（移到CONFIG定义之后）
tokenizer = BertTokenizer.from_pretrained(CONFIG["pretrained"])
model = BertForSequenceClassification.from_pretrained(CONFIG["pretrained"])

class MedicalDataset(Dataset):
    """医疗对话数据集加载与预处理类"""
    def __init__(self, json_path, tokenizer):
        # 读取JSON文件
        with open(json_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        
        # 数据清洗与转换
        self.samples = []
        for dialog in self.raw_data.values():
            for utterance in dialog:
                if utterance['dialogue_act']:  # 过滤测试集空标签
                    # 文本清洗：保留中文、数字、标点
                    cleaned_text = re.sub(r'[^\u4e00-\u9fa5，。！？、：；“”‘’（）《》【】0-9]', '', utterance['sentence'])
                    self.samples.append({
                        'text': cleaned_text.strip(),
                        'speaker': CONFIG["speaker_mapping"][utterance['speaker']],
                        'label': CONFIG["class_names"].index(utterance['dialogue_act'])
                    })
        
        # 初始化tokenizer
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """获取单条数据并进行编码"""
        sample = self.samples[idx]
        # BERT输入编码
        encoding = self.tokenizer.encode_plus(
            sample['text'],
            max_length=CONFIG["max_len"],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'speaker': torch.tensor(sample['speaker'], dtype=torch.long),
            'label': torch.tensor(sample['label'], dtype=torch.long)
        }

class MedicalBert(BertForSequenceClassification):
    """自定义医疗BERT模型（添加说话者信息）"""
    def __init__(self, config):
        super().__init__(config)
        # 添加说话者嵌入层
        self.speaker_emb = nn.Embedding(2, config.hidden_size)  # 医生/患者两种角色
        
        # 增强分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(config.hidden_size, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, len(CONFIG["class_names"]))
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, speaker_ids=None, **kwargs):
        # BERT基础编码
        outputs = self.bert(input_ids, 
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        
        # 融合说话者信息（添加空值检查）
        if speaker_ids is not None:
            speaker_emb = self.speaker_emb(speaker_ids)
            combined = pooled_output + speaker_emb
        else:
            combined = pooled_output
        
        # 分类预测
        logits = self.classifier(combined)
        return logits

def train_model():
    """模型训练主函数"""
    # 修改模型加载方式（新增镜像源参数）
    tokenizer = BertTokenizer.from_pretrained(
        CONFIG["pretrained"],
        mirror="https://hf-mirror.com"  # 新增镜像参数
    )
    
    # 加载数据集
    train_set = MedicalDataset(CONFIG["data_paths"]["train"], tokenizer)
    dev_set = MedicalDataset(CONFIG["data_paths"]["dev"], tokenizer)
    
    train_loader = DataLoader(train_set, batch_size=CONFIG["batch_size"], shuffle=True)
    dev_loader = DataLoader(dev_set, batch_size=CONFIG["batch_size"])
    
    # 设置分层学习率优化器
    optimizer = AdamW([
        {'params': model.bert.parameters(), 'lr': CONFIG["lr"]},
        {'params': model.classifier.parameters(), 'lr': CONFIG["lr"]*3}
    ])
    
    # 训练循环
    for epoch in range(CONFIG["epochs"]):
        model.train()
        for batch in train_loader:
            # 前向传播
            # 在train_model函数中：
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                speaker_ids=batch['speaker']  # 将speaker_ids单独传入
            )
            
            # 计算损失
            loss = torch.nn.CrossEntropyLoss()(outputs, batch['label'])
            
            # 反向传播
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # 验证步骤
        model.eval()
        total_acc = 0
        with torch.no_grad():
            for batch in dev_loader:
                # 修改前
                # outputs = model(**batch)
                
                # 修改后：显式传递参数并重命名
                outputs = model(
                    input_ids=batch['input_ids'],
                    speaker_ids=batch['speaker'],  # 关键修改点
                    attention_mask=batch['attention_mask']
                )
                preds = torch.argmax(outputs, dim=1)
                total_acc += (preds == batch['label']).sum().item()
        
        print(f"Epoch {epoch} | 验证集准确率: {total_acc/len(dev_set):.4f}")

def predict():
    """生成预测结果"""
    # 修改模型加载方式
    tokenizer = BertTokenizer.from_pretrained(
        CONFIG["pretrained"],
        mirror="https://hf-mirror.com"  # 新增镜像源参数
    )
    model = MedicalBert.from_pretrained(
        CONFIG["pretrained"],
        num_labels=len(CONFIG["class_names"]),
        mirror="https://hf-mirror.com"  # 添加镜像参数
    )
    
    # 读取测试数据
    with open(CONFIG["data_paths"]["test"], 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    predictions = {}
    model.eval()
    
    # 逐条预测
    for example_id, dialog in test_data.items():
        predictions[example_id] = []
        for utterance in dialog:
            # 数据预处理
            inputs = tokenizer(
                re.sub(r'[^\u4e00-\u9fa5，。！？、：；“”‘’（）《》【】0-9]', '', utterance['sentence']),
                max_length=CONFIG["max_len"],
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            speaker = CONFIG["speaker_mapping"][utterance['speaker']]
            
            # 模型预测
            with torch.no_grad():
                # 在predict函数中：
                outputs = model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    speaker_ids=torch.tensor([speaker])
                )
            
            # 保存预测结果
            pred = CONFIG["class_names"][torch.argmax(outputs).item()]
            predictions[example_id].append({
                **utterance,  # 移除多余的1，保持字典解构语法
                "dialogue_act": pred
            })
    
    # 保存结果文件
    with open(CONFIG["data_paths"]["output"], 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # 执行训练
    train_model()
    predict()