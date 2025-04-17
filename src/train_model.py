#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import logging
import argparse
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from transformers import BertConfig, BertTokenizer

# 导入自定义模块
from model_architecture import BertForDialogueActClassification, load_model, get_optimizer_and_scheduler
from model_architecture import DIALOGUE_ACTS, LABEL2ID, ID2LABEL
from data_processing import process_dataset_with_tokenization, tokenize_with_bert, clean_text, normalize_text

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class IMCSDataset(Dataset):
    """医疗对话意图分类数据集"""
    
    def __init__(self, data_file, tokenizer, max_length=512, is_training=True, context_size=2):
        """
        初始化数据集
        
        参数:
            data_file (str): 数据文件路径
            tokenizer (BertTokenizer): BERT分词器
            max_length (int): 最大序列长度
            is_training (bool): 是否为训练模式
            context_size (int): 上下文保留轮数
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_training = is_training
        self.context_size = context_size
        
        # 加载并预处理数据
        self.examples = []
        self.labels = []
        
        # 尝试加载预处理后的数据
        processed_file = data_file.replace('.json', '_tokenized.json')
        if os.path.exists(processed_file):
            logger.info(f"加载预处理数据: {processed_file}")
            self._load_processed_data(processed_file)
        else:
            logger.info(f"预处理数据: {data_file}")
            self._process_and_load_data(data_file)
    
    def _load_processed_data(self, processed_file):
        """加载预处理好的数据"""
        try:
            with open(processed_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for dialogue_id, dialogue in data.items():
                dialogue_context = []
                
                for utterance in dialogue:
                    # 获取标签
                    label = utterance.get('dialogue_act', 'Other')
                    label_id = LABEL2ID.get(label, 0)  # 默认为Other
                    
                    # 获取分词结果
                    tokenization = utterance.get('tokenization', None)
                    
                    if tokenization:
                        self.examples.append({
                            'input_ids': tokenization['input_ids'],
                            'attention_mask': tokenization['attention_mask'],
                            'token_type_ids': tokenization['token_type_ids'],
                            'sentence': utterance.get('sentence', ''),
                            'context': utterance.get('context_history', []),
                            'has_context': tokenization.get('has_context', False)
                        })
                        self.labels.append(label_id)
                    
                    # 将当前句子添加到对话上下文
                    dialogue_context.append(utterance.get('sentence', ''))
                    if len(dialogue_context) > self.context_size:
                        dialogue_context.pop(0)
            
            logger.info(f"已加载 {len(self.examples)} 个样本")
        
        except Exception as e:
            logger.error(f"加载预处理数据失败: {e}")
            raise
    
    def _process_and_load_data(self, data_file):
        """处理并加载原始数据"""
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for dialogue_id, dialogue in data.items():
                dialogue_context = []
                
                for utterance in dialogue:
                    # 获取标签
                    label = utterance.get('dialogue_act', 'Other')
                    label_id = LABEL2ID.get(label, 0)  # 默认为Other
                    
                    # 获取句子并处理
                    sentence = utterance.get('sentence', '')
                    cleaned_sentence = clean_text(sentence)
                    normalized_sentence = normalize_text(cleaned_sentence)
                    
                    # 进行分词
                    current_context = dialogue_context[-self.context_size:] if dialogue_context else []
                    tokenized = tokenize_with_bert(
                        normalized_sentence,
                        max_length=self.max_length,
                        tokenizer=self.tokenizer,
                        context=current_context,
                        context_size=self.context_size
                    )
                    
                    if tokenized:
                        self.examples.append({
                            'input_ids': tokenized['input_ids'],
                            'attention_mask': tokenized['attention_mask'],
                            'token_type_ids': tokenized['token_type_ids'],
                            'sentence': normalized_sentence,
                            'context': current_context,
                            'has_context': tokenized.get('has_context', False)
                        })
                        self.labels.append(label_id)
                    
                    # 将当前句子添加到对话上下文
                    dialogue_context.append(normalized_sentence)
                    if len(dialogue_context) > self.context_size:
                        dialogue_context.pop(0)
            
            logger.info(f"已处理 {len(self.examples)} 个样本")
        
        except Exception as e:
            logger.error(f"处理数据失败: {e}")
            raise
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        item = self.examples[idx]
        label = self.labels[idx]
        
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(item['token_type_ids'], dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def train(args):
    """训练模型"""
    # 设置随机种子
    set_seed(args.seed)
    
    # 检查是否有GPU可用
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载tokenizer
    try:
        tokenizer = BertTokenizer.from_pretrained(args.model_path)
        logger.info(f"已加载分词器: {args.model_path}")
    except:
        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        logger.info("已从在线加载分词器")
    
    # 加载数据集
    train_dataset = IMCSDataset(
        args.train_file,
        tokenizer,
        max_length=args.max_seq_length,
        is_training=True,
        context_size=args.context_size
    )
    
    dev_dataset = IMCSDataset(
        args.dev_file,
        tokenizer,
        max_length=args.max_seq_length,
        is_training=False,
        context_size=args.context_size
    )
    
    # 创建数据加载器
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size
    )
    
    dev_sampler = SequentialSampler(dev_dataset)
    dev_dataloader = DataLoader(
        dev_dataset,
        sampler=dev_sampler,
        batch_size=args.eval_batch_size
    )
    
    # 加载模型
    model = load_model(args.model_path, num_classes=len(DIALOGUE_ACTS))
    
    if model is None:
        logger.error("模型加载失败")
        return
    
    # 冻结BERT部分层参数
    if args.freeze_bert_layers > 0:
        # 获取所有BERT层
        bert_layers = list(model.bert.encoder.layer)
        num_layers = len(bert_layers)
        
        # 冻结前几层
        layers_to_freeze = min(args.freeze_bert_layers, num_layers)
        logger.info(f"冻结BERT前 {layers_to_freeze} 层")
        
        for i in range(layers_to_freeze):
            for param in bert_layers[i].parameters():
                param.requires_grad = False
    
    # 移动模型到设备
    model.to(device)
    
    # 获取优化器和学习率调度器
    optimizer, scheduler = get_optimizer_and_scheduler(
        model,
        train_dataloader,
        epochs=args.num_train_epochs,
        lr=args.learning_rate,
        warmup_ratio=args.warmup_ratio
    )
    
    # 训练前的统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型总参数数量: {total_params}")
    logger.info(f"可训练参数数量: {trainable_params} ({trainable_params/total_params*100:.2f}%)")
    
    # 记录训练历史
    training_stats = []
    best_f1 = 0.0
    
    # 训练循环
    logger.info("开始训练...")
    global_step = 0
    
    for epoch_i in range(args.num_train_epochs):
        # 训练阶段
        logger.info(f"{'='*20} Epoch {epoch_i + 1} / {args.num_train_epochs} {'='*20}")
        
        # 重置累计损失
        epoch_loss = 0.0
        model.train()
        
        # 显示进度条
        progress_bar = tqdm(train_dataloader, desc="训练")
        
        for step, batch in enumerate(progress_bar):
            # 将批次数据移动到设备
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 清零梯度
            model.zero_grad()
            
            # 前向传播
            outputs = model(**batch)
            loss = outputs['loss']
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            # 更新参数
            optimizer.step()
            scheduler.step()
            
            # 累计损失
            epoch_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({'loss': loss.item()})
            
            global_step += 1
        
        # 计算平均损失
        avg_train_loss = epoch_loss / len(train_dataloader)
        logger.info(f"平均训练损失: {avg_train_loss:.4f}")
        
        # 验证阶段
        logger.info("开始验证...")
        model.eval()
        
        val_accuracy = 0
        val_loss = 0
        all_preds = []
        all_labels = []
        
        # 不计算梯度
        with torch.no_grad():
            for batch in tqdm(dev_dataloader, desc="验证"):
                # 将批次数据移动到设备
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # 前向传播
                outputs = model(**batch)
                loss = outputs['loss']
                logits = outputs['logits']
                
                # 累计损失
                val_loss += loss.item()
                
                # 计算预测结果
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                labels = batch['labels'].cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels)
        
        # 计算准确率和F1分数
        val_accuracy = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        avg_val_loss = val_loss / len(dev_dataloader)
        
        logger.info(f"验证损失: {avg_val_loss:.4f}")
        logger.info(f"验证准确率: {val_accuracy:.4f}")
        logger.info(f"验证F1分数: {val_f1:.4f}")
        
        # 保存最佳模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            
            # 保存模型
            model_save_path = os.path.join(args.output_dir, f"best_model_epoch_{epoch_i+1}")
            model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            
            # 保存分类报告
            report = classification_report(
                all_labels, all_preds,
                target_names=DIALOGUE_ACTS,
                digits=4
            )
            with open(os.path.join(model_save_path, "classification_report.txt"), "w") as f:
                f.write(report)
            
            logger.info(f"已保存最佳模型到 {model_save_path}")
            logger.info(f"分类报告:\n{report}")
        
        # 记录训练统计信息
        training_stats.append({
            'epoch': epoch_i + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_accuracy': val_accuracy,
            'val_f1': val_f1
        })
    
    # 训练结束，绘制损失曲线
    plt.figure(figsize=(12, 6))
    plt.plot([stat['epoch'] for stat in training_stats], [stat['train_loss'] for stat in training_stats], label='训练损失')
    plt.plot([stat['epoch'] for stat in training_stats], [stat['val_loss'] for stat in training_stats], label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练和验证损失')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'loss_curve.png'))
    
    # 绘制准确率曲线
    plt.figure(figsize=(12, 6))
    plt.plot([stat['epoch'] for stat in training_stats], [stat['val_accuracy'] for stat in training_stats], label='准确率')
    plt.plot([stat['epoch'] for stat in training_stats], [stat['val_f1'] for stat in training_stats], label='F1分数')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('验证准确率和F1分数')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'accuracy_curve.png'))
    
    logger.info("训练完成")
    return training_stats


def predict(args):
    """预测并输出结果"""
    # 检查是否有GPU可用
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载模型和分词器
    try:
        model_path = args.model_path if args.predict_model_path is None else args.predict_model_path
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForDialogueActClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()
        logger.info(f"已加载模型: {model_path}")
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        return
    
    # 加载测试数据
    try:
        with open(args.test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        logger.info(f"已加载测试数据: {args.test_file}")
    except Exception as e:
        logger.error(f"加载测试数据失败: {e}")
        return
    
    # 初始化结果数据
    result_data = {}
    
    # 对每个对话进行处理
    for dialogue_id, dialogue in tqdm(test_data.items(), desc="预测对话"):
        processed_dialogue = []
        dialogue_context = []
        
        for utterance in dialogue:
            # 复制原始数据
            processed_utterance = utterance.copy()
            
            # 获取句子并处理
            sentence = utterance.get('sentence', '')
            cleaned_sentence = clean_text(sentence)
            normalized_sentence = normalize_text(cleaned_sentence)
            
            # 获取当前的上下文
            current_context = dialogue_context[-args.context_size:] if dialogue_context else []
            
            # 分词
            tokenized = tokenize_with_bert(
                normalized_sentence,
                max_length=args.max_seq_length,
                tokenizer=tokenizer,
                context=current_context,
                context_size=args.context_size
            )
            
            # 如果分词成功，进行预测
            if tokenized:
                # 准备输入数据
                input_data = {
                    'input_ids': torch.tensor([tokenized['input_ids']], dtype=torch.long).to(device),
                    'attention_mask': torch.tensor([tokenized['attention_mask']], dtype=torch.long).to(device),
                    'token_type_ids': torch.tensor([tokenized['token_type_ids']], dtype=torch.long).to(device)
                }
                
                # 预测
                with torch.no_grad():
                    outputs = model(**input_data)
                    logits = outputs['logits']
                    
                    # 获取预测结果
                    pred_id = torch.argmax(logits, dim=1).item()
                    pred_label = ID2LABEL.get(pred_id, 'Other')
                    
                    # 更新预测结果
                    processed_utterance['dialogue_act'] = pred_label
            
            # 更新对话上下文
            dialogue_context.append(normalized_sentence)
            if len(dialogue_context) > args.context_size:
                dialogue_context.pop(0)
            
            processed_dialogue.append(processed_utterance)
        
        result_data[dialogue_id] = processed_dialogue
    
    # 保存预测结果
    output_file = args.output_file if args.output_file else 'prediction_results.json'
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        logger.info(f"预测结果已保存至: {output_file}")
    except Exception as e:
        logger.error(f"保存预测结果失败: {e}")
    
    return result_data


def main():
    print("开始执行train_model.py脚本...")
    print("检查命令行参数...")
    
    parser = argparse.ArgumentParser(description="医疗对话意图分类模型训练与预测")
    
    # 通用参数
    parser.add_argument("--model_path", default="src/models/bert-base-chinese", type=str,
                        help="预训练模型路径")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="最大序列长度")
    parser.add_argument("--seed", default=42, type=int,
                        help="随机种子")
    parser.add_argument("--no_cuda", action="store_true",
                        help="不使用CUDA")
    parser.add_argument("--context_size", default=2, type=int,
                        help="上下文保留轮数")
    
    # 训练参数
    parser.add_argument("--do_train", action="store_true",
                        help="进行训练")
    parser.add_argument("--train_file", default="src/datasets/IMCS-DAC_train.json", type=str,
                        help="训练数据文件")
    parser.add_argument("--dev_file", default="src/datasets/IMCS-DAC_dev.json", type=str,
                        help="验证数据文件")
    parser.add_argument("--output_dir", default="src/models/fine-tuned", type=str,
                        help="输出目录")
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="训练批次大小")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="验证批次大小")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="学习率")
    parser.add_argument("--num_train_epochs", default=4, type=int,
                        help="训练轮数")
    parser.add_argument("--warmup_ratio", default=0.1, type=float,
                        help="预热步数比例")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="梯度裁剪阈值")
    parser.add_argument("--freeze_bert_layers", default=0, type=int,
                        help="冻结BERT的前几层，0表示不冻结")
    
    # 预测参数
    parser.add_argument("--do_predict", action="store_true",
                        help="进行预测")
    parser.add_argument("--test_file", default="src/datasets/IMCS-DAC_test.json", type=str,
                        help="测试数据文件")
    parser.add_argument("--predict_model_path", default=None, type=str,
                        help="预测使用的模型路径，不指定则使用model_path")
    parser.add_argument("--output_file", default=None, type=str,
                        help="预测结果输出文件")
    
    args = parser.parse_args()
    print(f"解析到的参数: {args}")
    
    # 检查是否指定了任务
    if not args.do_train and not args.do_predict:
        print("警告: 未指定任何任务(--do_train或--do_predict)，脚本将退出")
        parser.print_help()
        return
    
    # 路径检查与创建
    if args.do_train and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"创建输出目录: {args.output_dir}")
    
    # 检查数据文件
    if args.do_train:
        if not os.path.exists(args.train_file):
            print(f"错误: 训练文件不存在: {args.train_file}")
            return
        if not os.path.exists(args.dev_file):
            print(f"错误: 验证文件不存在: {args.dev_file}")
            return
        print(f"训练文件检查通过: {args.train_file}")
        print(f"验证文件检查通过: {args.dev_file}")
    
    if args.do_predict:
        if not os.path.exists(args.test_file):
            print(f"错误: 测试文件不存在: {args.test_file}")
            return
        print(f"测试文件检查通过: {args.test_file}")
    
    # 检查模型路径
    if not os.path.exists(args.model_path):
        print(f"警告: 模型路径不存在: {args.model_path}")
        print("将尝试从HuggingFace在线加载模型")
    else:
        print(f"模型路径检查通过: {args.model_path}")
    
    # 执行训练
    if args.do_train:
        print("准备开始训练流程...")
        logger.info("开始训练流程")
        train(args)
    
    # 执行预测
    if args.do_predict:
        print("准备开始预测流程...")
        logger.info("开始预测流程")
        predict(args)
    
    print("脚本执行完毕")


if __name__ == "__main__":
    main()
