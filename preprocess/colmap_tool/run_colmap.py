import os
import subprocess
import argparse

def run_colmap_commands(base_dir):
    """
    执行 COLMAP 处理流程的 Python 封装
    Args:
        base_dir: 基础目录路径 (如 "dataset/occlusion/clothing")
    """
    os.chdir(base_dir)
    print(f"当前工作目录: {os.getcwd()}")

    print("开始特征提取...")
    subprocess.run([
        "colmap", "feature_extractor",
        "--database_path", "database.db",
        "--image_path", "images/"
    ], check=True)

    print("开始特征匹配...")
    subprocess.run([
        "colmap", "exhaustive_matcher",
        "--database_path", "database.db"
    ], check=True)

    print("开始稀疏重建...")
    os.makedirs("sparse", exist_ok=True)
    subprocess.run([
        "colmap", "mapper",
        "--database_path", "database.db",
        "--image_path", "images/",
        "--output_path", "sparse/"
    ], check=True)

    print("转换参数为文本格式...")
    sparse_dir = os.path.join("sparse", "0")
    if os.path.exists(sparse_dir):
        subprocess.run([
            "colmap", "model_converter",
            "--input_path", sparse_dir,
            "--output_path", sparse_dir,
            "--output_type", "TXT"
        ], check=True)
    else:
        print(f"警告: {sparse_dir} 不存在，跳过模型转换")

    print("COLMAP 处理完成")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行 COLMAP 三维重建流程")
    parser.add_argument("--base_dir", type=str, required=True,
                        help="基础目录路径 (例如 'dataset/occlusion/clothing')")
    args = parser.parse_args()

    try:
        run_colmap_commands(args.base_dir)
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败: {e}")
    except Exception as e:
        print(f"发生错误: {str(e)}")