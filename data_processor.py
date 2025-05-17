#数据字典加载
import json
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
import jieba
import random
from collections import Counter

class TextAugmenter:
    def __init__(self):
        # 加载同义词词典
        self.synonyms = self._load_synonyms()
        # 加载停用词
        self.stopwords = self._load_stopwords()
        
    def _load_synonyms(self):
        # 这里可以加载自定义的同义词词典
        # 示例同义词词典
        return {
            '头痛': ['头疼', '头部疼痛', '头不舒服'],
            '发烧': ['发热', '体温升高', '发高烧'],
            '咳嗽': ['咳', '干咳', '咳痰'],
            '恶心': ['想吐', '反胃', '呕吐感'],
            '疲劳': ['累', '疲倦', '乏力']
        }
        
    def _load_stopwords(self):
        # 加载停用词表
        return set(['的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'])
        
    def synonym_replacement(self, text, n=1):
        words = list(jieba.cut(text))
        new_words = words.copy()
        random_word_list = list(set([word for word in words if word not in self.stopwords]))
        random.shuffle(random_word_list)
        num_replaced = 0
        
        for random_word in random_word_list:
            if random_word in self.synonyms:
                synonym = random.choice(self.synonyms[random_word])
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1
                if num_replaced >= n:
                    break
                    
        return ''.join(new_words)
        
    def random_insertion(self, text, n=1):
        words = list(jieba.cut(text))
        new_words = words.copy()
        for _ in range(n):
            word = random.choice(new_words)
            if word not in self.stopwords:
                new_words.insert(random.randint(0, len(new_words)), word)
        return ''.join(new_words)
        
    def augment(self, text):
        """
        对文本进行数据增强
        
        Args:
            text (str): 原始文本
            
        Returns:
            str: 增强后的文本
        """
        # 随机选择一种增强方法
        methods = [self.synonym_replacement, self.random_insertion]
        method = random.choice(methods)
        
        # 随机选择增强次数（1-2次）
        n = random.randint(1, 2)
        
        # 应用增强
        augmented_text = method(text, n)
        
        return augmented_text

class MedicalIntentDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128, augment=False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        self.augmenter = TextAugmenter() if augment else None
        
        # 创建标签映射
        self.label2id = {label: idx for idx, label in enumerate(sorted(set(item['label'] for item in data)))}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label = self.label2id[item['label']]  # 将标签转换为ID
        
        # 如果启用数据增强，对训练数据进行增强
        if self.augment and self.augmenter:
            if random.random() < 0.5:  # 50%的概率进行增强
                text = self.augmenter.augment(text)
        
        # 对文本进行编码
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data(file_path):
    """
    加载IMCS-DAC数据集
    
    Args:
        file_path (str): 数据文件路径
        
    Returns:
        list: 处理后的数据列表，每个元素包含text和label
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 处理IMCS-DAC数据集格式
    processed_data = []
    for dialogue_id, dialogue in data.items():
        for turn in dialogue:
            if turn.get('dialogue_act'):  # 只处理有标签的对话轮次
                processed_data.append({
                    'text': turn['sentence'],
                    'label': turn['dialogue_act']
                })
    
    return processed_data

def create_data_loaders(train_data, val_data, test_data, tokenizer, batch_size=32, max_length=128):
    # 创建数据集
    train_dataset = MedicalIntentDataset(train_data, tokenizer, max_length, augment=True)
    val_dataset = MedicalIntentDataset(val_data, tokenizer, max_length)
    test_dataset = MedicalIntentDataset(test_data, tokenizer, max_length)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 获取标签数量
    num_labels = len(set(item['label'] for item in train_data))
    
    return train_loader, val_loader, test_loader, num_labels 
