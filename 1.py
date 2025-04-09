import os
import argparse

def add_gitkeep_to_empty_dirs(root_dir):
    """
    递归遍历目录，向所有空文件夹添加 .gitkeep 文件
    
    Args:
        root_dir (str): 要遍历的根目录路径
    """
    count = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 检查当前目录是否为空（没有文件和子目录）
        if not dirnames and not filenames:
            gitkeep_path = os.path.join(dirpath, '.gitkeep')
            with open(gitkeep_path, 'w') as f:
                pass  # 创建空文件
            print(f"已添加: {gitkeep_path}")
            count += 1
        # 检查当前目录是否只包含 .git 目录
        elif len(dirnames) == 1 and '.git' in dirnames and not filenames:
            gitkeep_path = os.path.join(dirpath, '.gitkeep')
            with open(gitkeep_path, 'w') as f:
                pass
            print(f"已添加: {gitkeep_path}")
            count += 1
    
    print(f"\n完成! 共向 {count} 个空文件夹添加了 .gitkeep 文件")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='向所有空文件夹添加 .gitkeep 文件')
    parser.add_argument('--dir', type=str, default='.', 
                        help='要处理的根目录路径 (默认为当前目录)')
    args = parser.parse_args()
    
    # 获取绝对路径
    root_dir = os.path.abspath(args.dir)
    print(f"开始处理目录: {root_dir}\n")
    
    add_gitkeep_to_empty_dirs(root_dir)