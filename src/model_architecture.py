# 模型架构

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel, BertConfig, AdamW, get_linear_schedule_with_warmup


class BertForDialogueActClassification(BertPreTrainedModel):
    """
    基于BERT的医疗对话意图分类模型
    
    扩展BERT预训练模型，增加下游任务的分类层
    """
    
    def __init__(self, config, num_classes=16):
        super().__init__(config)
        self.num_labels = num_classes
        
        # 加载BERT模型
        self.bert = BertModel(config)
        
        # 分类头，取[CLS]标记的输出向量进行分类
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_classes)
        
        # 初始化权重
        self.init_weights()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        前向传播函数
        
        参数:
            input_ids: 输入序列的token IDs
            attention_mask: 注意力掩码
            token_type_ids: token类型IDs
            position_ids: 位置编码
            labels: 真实标签
            
        返回:
            loss: 如果提供labels，则返回损失值
            logits: 模型的输出logits
        """
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 通过BERT模型获取输出
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # 获取[CLS]标记位置的隐藏状态
        pooled_output = outputs[1]  # [CLS]的输出
        
        # 应用dropout并通过分类器
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            # 计算交叉熵损失
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
        }


def load_model(model_path='src/models/bert-base-chinese', num_classes=16):
    """
    加载预训练的BERT模型，并添加分类层
    
    参数:
        model_path: BERT预训练模型路径
        num_classes: 分类标签数量
        
    返回:
        model: 加载好的模型
    """
    try:
        # 加载BERT配置
        config = BertConfig.from_pretrained(
            model_path,
            num_labels=num_classes,
            output_attentions=False,
            output_hidden_states=False,
        )
        
        # 创建模型
        model = BertForDialogueActClassification.from_pretrained(
            model_path,
            config=config,
            num_classes=num_classes
        )
        
        return model
    
    except Exception as e:
        print(f"加载模型失败: {e}")
        # 如果本地加载失败，尝试从Hugging Face在线加载
        try:
            config = BertConfig.from_pretrained(
                "bert-base-chinese",
                num_labels=num_classes,
                output_attentions=False,
                output_hidden_states=False,
            )
            
            model = BertForDialogueActClassification.from_pretrained(
                "bert-base-chinese",
                config=config,
                num_classes=num_classes
            )
            
            return model
        
        except Exception as e:
            print(f"在线加载模型失败: {e}")
            return None


def get_optimizer_and_scheduler(model, train_dataloader, epochs=5, lr=2e-5, warmup_ratio=0.1):
    """
    创建优化器和学习率调度器
    
    参数:
        model: 模型
        train_dataloader: 训练数据加载器
        epochs: 训练轮数
        lr: 学习率
        warmup_ratio: warmup比例
        
    返回:
        optimizer: AdamW优化器
        scheduler: 学习率调度器
    """
    # 准备优化器参数
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    
    # 创建AdamW优化器
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)
    
    # 计算总训练步数
    total_steps = len(train_dataloader) * epochs
    
    # 创建学习率调度器，带warmup策略
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    return optimizer, scheduler


# 对话意图标签
DIALOGUE_ACTS = [
    'Other',
    'Request-Symptom',
    'Inform-Symptom',
    'Request-Basic_Information',
    'Inform-Basic_Information',
    'Request-Etiology',
    'Diagnose',
    'Inform-Precautions',
    'Request-Existing_Examination_and_Treatment',
    'Inform-Existing_Examination_and_Treatment',
    'Request-Drug_Recommendation',
    'Inform-Drug_Recommendation',
    'Request-Examination_and_Treatment',
    'Inform-Examination_and_Treatment',
    'Request-Healthcare_Institution',
    'Inform-Healthcare_Institution'
]

# 创建标签到索引的映射
LABEL2ID = {label: i for i, label in enumerate(DIALOGUE_ACTS)}
ID2LABEL = {i: label for i, label in enumerate(DIALOGUE_ACTS)}

