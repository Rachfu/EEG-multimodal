import os
# 定义需要处理的文件夹路径
root_folder = 'python/logs/'
# 遍历根文件夹中的所有子文件夹和文件
for subdir, _, files in os.walk(root_folder):
    for file_name in files:
        # 检查文件是否是txt文件
        if file_name.endswith('.txt'):
            file_path = os.path.join(subdir, file_name)
            
            # 打开文件并读取内容
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            
            # 修改每一行的缩进和替换文本
            new_lines = []
            for line in lines:
                # 替换"Val"为"Test"
                line = line.replace('Val', 'Test')
                
                # 检查是否有'|'并调整缩进
                if line.startswith(' ' * 12 + '|'):
                    # 替换前面的12个空格为16个空格
                    new_line = line.replace(' ' * 12 + '|', ' ' * 16 + '|', 1)
                    new_lines.append(new_line)
                else:
                    new_lines.append(line)
            
            # 将修改后的内容写回文件
            with open(file_path, 'w', encoding='utf-8') as file:
                file.writelines(new_lines)
print("处理完成：缩进调整和文本替换已完成！")