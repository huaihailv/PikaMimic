import os
import shutil

def rename_episodes(directory1, output_directory):
    # 获取两个目录下的所有HDF5文件
    files1 = [os.path.join(directory1, f) for f in os.listdir(directory1) if f.endswith('.h5')]
    # files2 = [os.path.join(directory2, f) for f in os.listdir(directory2) if f.endswith('.h5')]
    
    # 合并两个目录的文件列表
    all_files = files1 
    
    # 创建输出目录（如果不存在）
    os.makedirs(output_directory, exist_ok=True)
    
    # 重命名并复制文件
    for index, file_path in enumerate(all_files):
        new_file_name = f"episode_{index}.hdf5"
        new_file_path = os.path.join(output_directory, new_file_name)
        shutil.copy(file_path, new_file_path)
        print(f"Copied {file_path} to {new_file_path}")

if __name__ == "__main__":
    # 指定两个输入目录和一个输出目录
    directory1 = "/home/lvhuaihai/data_20250122-163942"
    # directory2 = "/home/lvhuaihai/data_20250116-154250"
    output_directory = "/home/lvhuaihai/data_pour3cups_video"
    
    rename_episodes(directory1, output_directory)
    