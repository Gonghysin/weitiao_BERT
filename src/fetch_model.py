# 从huggingface下载模型
import os
from transformers import AutoTokenizer, AutoModel

def download_bert_model():
    """
    从Hugging Face下载bert-base-chinese模型到src/models目录
    """
    model_name = "bert-base-chinese"
    save_directory = os.path.join(os.path.dirname(__file__), "models", model_name)
    
    # 创建保存目录
    os.makedirs(save_directory, exist_ok=True)
    
    print(f"开始下载 {model_name} 模型...")
    
    # 下载并保存tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_directory)
    
    # 下载并保存模型
    model = AutoModel.from_pretrained(model_name)
    model.save_pretrained(save_directory)
    
    print(f"模型已成功下载并保存到 {save_directory}")

if __name__ == "__main__":
    download_bert_model()

