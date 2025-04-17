#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import re
import os
import logging
from zhconv import convert
import unicodedata
from transformers import BertTokenizer
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# BERT模型路径
BERT_MODEL_PATH = 'src/models/bert-base-chinese'

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

def tokenize_with_bert(text, max_length=512, preserve_context=True, tokenizer=None):
    """
    使用BERT分词器对文本进行分词处理
    
    参数:
        text (str): 需要分词的文本
        max_length (int): 最大序列长度，默认为512
        preserve_context (bool): 是否保留上下文，默认为True
        tokenizer (BertTokenizer): 可选的已加载的分词器实例
        
    返回:
        dict: 包含tokenized结果的字典，包括:
            - input_ids: token对应的ID
            - token_type_ids: 段落ID
            - attention_mask: 注意力掩码
            - tokens: 分词后的token列表
    """
    if not text or not isinstance(text, str):
        return None
    
    try:
        # 如果没有提供tokenizer，则加载BERT分词器
        if tokenizer is None:
            try:
                tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
                logger.info(f"已加载BERT分词器: {BERT_MODEL_PATH}")
            except Exception as e:
                logger.error(f"加载本地BERT分词器失败: {e}")
                # 尝试从在线加载
                try:
                    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
                    logger.info("已从在线资源加载BERT分词器")
                except Exception as e:
                    logger.error(f"加载BERT分词器失败: {e}")
                    return None
        
        # 对文本进行分词
        encoding = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding='max_length' if max_length else False,
            return_tensors='np'
        )
        
        # 将numpy数组转换为普通列表以便于序列化
        result = {
            'input_ids': encoding['input_ids'].tolist()[0],
            'token_type_ids': encoding['token_type_ids'].tolist()[0],
            'attention_mask': encoding['attention_mask'].tolist()[0]
        }
        
        # 获取分词后的token列表
        tokens = tokenizer.convert_ids_to_tokens(result['input_ids'])
        result['tokens'] = tokens
        
        return result
    
    except Exception as e:
        logger.error(f"BERT分词处理失败: {e}")
        return None

def filter_stopwords(tokens, stopwords_file=None):
    """
    基于停用词表过滤分词结果
    
    参数:
        tokens (list): 分词后的token列表
        stopwords_file (str): 停用词表文件路径，如果为None则使用默认路径
        
    返回:
        list: 过滤后的token列表
    """
    if not tokens or not isinstance(tokens, list):
        return []
    
    try:
        # 如果没有提供停用词文件，使用默认路径
        if stopwords_file is None:
            stopwords_file = 'src/datasets/stopwords/chinese_stopwords.txt'
        
        # 加载停用词表
        stopwords = set()
        try:
            if os.path.exists(stopwords_file):
                with open(stopwords_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        word = line.strip()
                        if word:
                            stopwords.add(word)
                logger.info(f"已加载停用词表，共 {len(stopwords)} 个停用词")
            else:
                logger.warning(f"停用词表文件不存在: {stopwords_file}")
                # 创建一个小的默认停用词表
                stopwords = {'的', '了', '和', '是', '在', '我', '有', '就', '不', '也', '这', '上', '中', '大', '为', '来', '个', '地', '还', '子'}
                logger.info(f"使用默认停用词表，共 {len(stopwords)} 个停用词")
        except Exception as e:
            logger.error(f"加载停用词表失败: {e}")
            return tokens
        
        # 过滤停用词
        filtered_tokens = [token for token in tokens if token not in stopwords and not (token.startswith('##') and token[2:] in stopwords)]
        
        return filtered_tokens
    
    except Exception as e:
        logger.error(f"停用词过滤失败: {e}")
        return tokens

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

def process_dataset_with_tokenization(input_file, output_file=None, to_simplified=True, 
                                      apply_stopwords=False, max_length=512):
    """
    处理IMCS-DAC数据集并进行分词处理
    
    参数:
        input_file (str): 输入文件路径
        output_file (str): 输出文件路径，默认为None
        to_simplified (bool): 是否转换为简体中文，默认为True
        apply_stopwords (bool): 是否应用停用词过滤，默认为False
        max_length (int): 最大序列长度，默认为512
        
    返回:
        dict: 处理后的数据
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"加载数据文件失败: {e}")
        return None
    
    # 加载BERT分词器
    try:
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
        logger.info(f"已加载BERT分词器: {BERT_MODEL_PATH}")
    except Exception as e:
        logger.error(f"加载BERT分词器失败: {e}")
        try:
            tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
            logger.info("已从在线资源加载BERT分词器")
        except Exception as e:
            logger.error(f"加载BERT分词器失败，无法继续进行分词: {e}")
            return None
    
    processed_data = {}
    
    for dialogue_id, dialogue in data.items():
        processed_dialogue = []
        
        # 处理整个对话
        for utterance in dialogue:
            # 深拷贝utterance字典
            processed_utterance = utterance.copy()
            
            # 清洗和规范化句子
            sentence = utterance.get('sentence', '')
            cleaned_sentence = clean_text(sentence)
            normalized_sentence = normalize_text(cleaned_sentence, to_simplified)
            
            # BERT分词处理
            tokenization_result = tokenize_with_bert(
                normalized_sentence, 
                max_length=max_length,
                tokenizer=tokenizer
            )
            
            # 如果需要，应用停用词过滤
            if apply_stopwords and tokenization_result:
                filtered_tokens = filter_stopwords(tokenization_result['tokens'])
                tokenization_result['filtered_tokens'] = filtered_tokens
            
            processed_utterance['sentence'] = normalized_sentence
            if tokenization_result:
                processed_utterance['tokenization'] = tokenization_result
            
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
