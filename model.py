# 导入必要的库
import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块
from transformers import BertModel, BertTokenizer  # BERT预训练模型
from utils.medical_tools import MedicalIntentAnalyzer

# 定义注意力机制类
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, hidden_states):
        attention_weights = self.attention(hidden_states)
        attention_weights = self.dropout(attention_weights)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * hidden_states, dim=1)
        return context

# 定义医疗意图分类器模型
class MedicalIntentClassifier(nn.Module):
    def __init__(self, num_labels, model_name="bert-base-chinese", lstm_hidden_size=256, lstm_layers=2):
        super(MedicalIntentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.medical_analyzer = MedicalIntentAnalyzer()
        
        # 增加LSTM的dropout
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if lstm_layers > 1 else 0
        )
        
        # 增加层归一化
        self.layer_norm = nn.LayerNorm(lstm_hidden_size * 2)
        
        self.attention = Attention(lstm_hidden_size * 2)
        
        # 医疗特征处理层
        self.medical_feature_processor = nn.Sequential(
            nn.Linear(3, lstm_hidden_size),  # 3个主要意图分数
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.LayerNorm(lstm_hidden_size)
        )
        
        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(lstm_hidden_size * 3, lstm_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.LayerNorm(lstm_hidden_size)
        )
        
        self.classifier = nn.Linear(lstm_hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        # BERT编码
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs[0]
        
        # LSTM处理
        lstm_output, _ = self.lstm(sequence_output)
        lstm_output = self.layer_norm(lstm_output)
        
        # 注意力机制
        context = self.attention(lstm_output)
        
        # 处理医疗特征
        batch_size = input_ids.size(0)
        medical_features = torch.zeros(batch_size, 3).to(input_ids.device)
        
        for i in range(batch_size):
            text = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)
            intent_scores = self.medical_analyzer.analyze_intent(text)
            medical_features[i, 0] = intent_scores.get('symptom_description', 0)
            medical_features[i, 1] = intent_scores.get('disease_inquiry', 0)
            medical_features[i, 2] = intent_scores.get('treatment_advice', 0)
        
        medical_features = self.medical_feature_processor(medical_features)
        
        # 特征融合
        combined_features = torch.cat([context, medical_features], dim=1)
        combined_features = self.feature_fusion(combined_features)
        
        # 分类
        logits = self.classifier(combined_features)
        
        return logits