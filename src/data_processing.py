#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import re
import os
import logging
from zhconv import convert
import unicodedata

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_text(text):
    """
    文本清洗函数，移除噪音和冗余信息
    
    参数:
        text (str): 需要清洗的文本
    
    返回:
        str: 清洗后的文本
    """
    if not text or not isinstance(text, str):
        return ""
    
    # 移除HTML标签
    text = re.sub(r'<.*?>', '', text)
    
    # 移除多余的空白字符和换行符
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # 移除多余的特殊符号
    text = re.sub(r'([!?.,;:，。！？；：])\1+', r'\1', text)
    
    # 检查非中文字符和乱码
    # 保留中文、英文、数字、常用标点
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9!?,.:;，。！？、；：\s]', '', text)
    
    return text

def normalize_text(text, to_simplified=True):
    """
    规范化处理函数，统一符号和中文字体
    
    参数:
        text (str): 需要规范化的文本
        to_simplified (bool): 是否转换为简体中文，默认为True
    
    返回:
        str: 规范化后的文本
    """
    if not text or not isinstance(text, str):
        return ""
    
    # 全角转半角
    text = ''.join([unicodedata.normalize('NFKC', char) if ord(char) > 127 else char for char in text])
    
    # 标点符号统一 (统一使用中文标点)
    punctuation_map = {
        ',': '，',
        '.': '。',
        '!': '！',
        '?': '？',
        ':': '：',
        ';': '；',
        '(': '（',
        ')': '）',
        '[': '【',
        ']': '】'
    }
    
    for en_punct, cn_punct in punctuation_map.items():
        text = text.replace(en_punct, cn_punct)
    
    # 繁体转简体或简体转繁体
    if to_simplified:
        text = convert(text, 'zh-hans')
    else:
        text = convert(text, 'zh-hant')
    
    return text

def process_dataset(input_file, output_file=None, to_simplified=True):
    """
    处理IMCS-DAC数据集
    
    参数:
        input_file (str): 输入文件路径
        output_file (str): 输出文件路径，默认为None
        to_simplified (bool): 是否转换为简体中文，默认为True
        
    返回:
        dict: 处理后的数据
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"加载数据文件失败: {e}")
        return None
    
    processed_data = {}
    
    for dialogue_id, dialogue in data.items():
        processed_dialogue = []
        
        for utterance in dialogue:
            # 深拷贝utterance字典
            processed_utterance = utterance.copy()
            
            # 清洗和规范化句子
            sentence = utterance.get('sentence', '')
            cleaned_sentence = clean_text(sentence)
            normalized_sentence = normalize_text(cleaned_sentence, to_simplified)
            
            processed_utterance['sentence'] = normalized_sentence
            processed_dialogue.append(processed_utterance)
        
        processed_data[dialogue_id] = processed_dialogue
    
    # 保存处理后的数据
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
            logger.info(f"处理后的数据已保存至: {output_file}")
        except Exception as e:
            logger.error(f"保存数据文件失败: {e}")
    
    return processed_data

def main():
    # 数据集路径
    input_file = 'src/datasets/IMCS-DAC_train.json'
    output_file = 'src/datasets/IMCS-DAC_train_processed.json'
    
    # 处理数据集
    processed_data = process_dataset(input_file, output_file)
    
    if processed_data:
        # 输出处理统计信息
        dialogue_count = len(processed_data)
        utterance_count = sum(len(dialogue) for dialogue in processed_data.values())
        
        logger.info(f"处理完成: 共处理 {dialogue_count} 个对话, {utterance_count} 个句子")

if __name__ == "__main__":
    main()
