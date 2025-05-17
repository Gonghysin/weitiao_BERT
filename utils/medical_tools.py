import re
import json
from collections import defaultdict
import jieba
import jieba.posseg as pseg

class MedicalIntentAnalyzer:
    def __init__(self):
        # 加载医疗资源文件
        self._load_medical_terms('/root/weitiao_BERT--/data/medical_terms.json')
        self._load_symptom_patterns('/root/weitiao_BERT--/data/symptom_patterns.json')
        
        # 初始化分词词典
        jieba.load_userdict('/root/weitiao_BERT--/data/medical_dict.txt')
        self.stopwords = set(open('/root/weitiao_BERT--/data/chinese_stopwords.txt').read().splitlines())

    def _load_medical_terms(self, path):
        # 加载医学术语库
        with open(path) as f:
            data = json.load(f)
            self.diseases = set(data['diseases'])
            self.drugs = set(data['drugs'])
            self.procedures = set(data['procedures'])
            
    def _load_symptom_patterns(self, path):
        # 加载症状匹配规则
        with open(path) as f:
            self.symptom_patterns = json.load(f)
            
    def _extract_symptoms(self, text):
        # 使用规则和词典提取症状实体
        symptoms = []
        
        # 正则匹配
        for pattern in self.symptom_patterns.get('regex', []):
            matches = re.findall(pattern, text)
            symptoms.extend(matches)
            
        # 分词匹配
        words = pseg.cut(text)
        for word, flag in words:
            if flag == 'n' and word in self.symptom_patterns.get('nouns', []):
                symptoms.append(word)
                
        return list(set(symptoms))

    def analyze_intent(self, text):
        """
        分析医疗文本的核心意图特征
        返回包含以下特征的结构化数据：
        - symptom_score: 症状描述强度 (0-1)
        - inquiry_score: 疾病咨询强度 (0-1)
        - treatment_score: 治疗建议强度 (0-1)
        - symptom_entities: 识别到的症状实体列表
        - medical_terms: 包含的医学术语统计
        """
        # 特征初始化
        features = {
            'symptom_score': 0.0,
            'inquiry_score': 0.0,
            'treatment_score': 0.0,
            'symptom_entities': [],
            'medical_terms': defaultdict(int)
        }
        
        # 提取症状实体
        symptoms = self._extract_symptoms(text)
        features['symptom_entities'] = symptoms
        
        # 计算症状得分
        symptom_words = len([w for w in jieba.lcut(text) if w in self.symptom_patterns['keywords']['symptom']])
        features['symptom_score'] = min(0.3 * len(symptoms) + 0.1 * symptom_words, 1.0)
        
        # 分析疑问词
        inquiry_terms = ['吗', '是不是', '会不会', '如何', '怎样', '为什么']
        features['inquiry_score'] = 0.5 if any(term in text for term in inquiry_terms) else 0.0
        
        # 识别治疗相关术语
        treatment_keywords = ['建议', '治疗', '用药', '手术', '方案']
        drug_count = len([w for w in jieba.lcut(text) if w in self.drugs])
        procedure_count = len([w for w in jieba.lcut(text) if w in self.procedures])
        
        features['treatment_score'] = min(0.2 * drug_count + 0.3 * procedure_count, 1.0)
        features['medical_terms']['drug'] = drug_count
        features['medical_terms']['procedure'] = procedure_count
        
        return features

# 示例用法
if __name__ == "__main__":
    analyzer = MedicalIntentAnalyzer()
    sample_text = "我最近头痛得厉害，伴有发热，是不是感冒了？需要吃布洛芬吗？"
    print(analyzer.analyze_intent(sample_text))