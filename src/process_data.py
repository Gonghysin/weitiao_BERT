import json
import pandas as pd
import os
from collections import OrderedDict

def process_json_to_csv(json_file, output_csv, class_csv=None):
    """
    处理JSON文件并生成CSV文件
    
    Args:
        json_file: JSON文件路径
        output_csv: 输出CSV文件路径
        class_csv: 类别CSV文件路径，如果为None则不生成
    """
    # 读取JSON文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取所有对话数据
    all_dialogues = []
    dialogue_acts = set()
    
    for dialogue_id, dialogue in data.items():
        for utterance in dialogue:
            all_dialogues.append({
                'speaker': utterance['speaker'],
                'sentence': utterance['sentence'],
                'dialogue_act': utterance.get('dialogue_act', '')  # 测试集可能没有dialogue_act
            })
            if 'dialogue_act' in utterance:
                dialogue_acts.add(utterance['dialogue_act'])
    
    # 创建DataFrame并保存为CSV
    df = pd.DataFrame(all_dialogues)
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"已生成 {output_csv}")
    
    # 如果需要生成类别CSV
    if class_csv:
        # 按字符串大小排序
        sorted_acts = sorted(list(dialogue_acts))
        # 创建类别DataFrame
        class_df = pd.DataFrame({
            'id': range(len(sorted_acts)),
            'dialogue_act': sorted_acts
        })
        class_df.to_csv(class_csv, index=False, encoding='utf-8')
        print(f"已生成 {class_csv}")
    
    return dialogue_acts

def main():
    # 设置路径
    datasets_dir = os.path.join('src', 'datasets')
    train_json = os.path.join(datasets_dir, 'IMCS-DAC_train.json')
    dev_json = os.path.join(datasets_dir, 'IMCS-DAC_dev.json')
    test_json = os.path.join(datasets_dir, 'IMCS-DAC_test.json')
    
    train_csv = os.path.join(datasets_dir, 'IMCS-DAC_train.csv')
    dev_csv = os.path.join(datasets_dir, 'IMCS-DAC_dev.csv')
    test_csv = os.path.join(datasets_dir, 'IMCS-DAC_test.csv')
    class_csv = os.path.join(datasets_dir, 'class.csv')
    
    # 处理训练集并生成类别CSV
    process_json_to_csv(train_json, train_csv, class_csv)
    
    # 处理开发集
    process_json_to_csv(dev_json, dev_csv)
    
    # 处理测试集
    process_json_to_csv(test_json, test_csv)

if __name__ == "__main__":
    main()
