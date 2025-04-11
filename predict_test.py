import torch
from transformers import BertTokenizer
from model import MedicalIntentClassifier
from data_processor import MedicalIntentDataset
import json
from tqdm import tqdm

def load_model(model_path, num_labels):
    """加载训练好的模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MedicalIntentClassifier(num_labels=num_labels)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

def predict_intent(text, model, tokenizer, device, max_length=128):
    """预测单个文本的意图"""
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, predicted = torch.max(outputs, dim=1)
    
    predicted_label = MedicalIntentDataset.id2label[predicted.item()]
    return predicted_label

def main():
    # 加载标签映射
    with open('data/IMCS-DAC_train.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    # 获取所有可能的标签
    all_labels = set()
    for dialogue in train_data.values():
        for turn in dialogue:
            if turn['dialogue_act']:
                all_labels.add(turn['dialogue_act'])
    
    # 创建标签到ID的映射
    label2id = {label: idx for idx, label in enumerate(sorted(all_labels))}
    id2label = {idx: label for label, idx in label2id.items()}
    
    # 将标签映射添加到数据集类
    MedicalIntentDataset.label2id = label2id
    MedicalIntentDataset.id2label = id2label
    
    # 加载模型和tokenizer
    model, device = load_model('best_model.pth', len(label2id))
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    # 加载测试数据
    with open('data/IMCS-DAC_test.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # 预测并更新测试数据
    print("开始预测测试集中的对话意图...")
    for dialogue_id, dialogue in tqdm(test_data.items()):
        for turn in dialogue:
            if not turn['dialogue_act']:  # 只预测空标签
                text = turn['sentence']
                predicted_label = predict_intent(text, model, tokenizer, device)
                turn['dialogue_act'] = predicted_label
    
    # 保存更新后的测试数据
    output_file = 'data/IMCS-DAC_test_predicted.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n预测完成！结果已保存到: {output_file}")

if __name__ == '__main__':
    main() 
