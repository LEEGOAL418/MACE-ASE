import matplotlib.pyplot as plt
import re
import argparse
import os


def plot_log_data(log_file):
    # 初始化列表来存储每一轮的数据
    epochs = []
    loss_values = []
    rmse_e_values = []
    rmse_f_values = []

    # 使用正则表达式提取每一行中的loss、RMSE_E_per_atom和RMSE_F值
    pattern = r"Epoch (\d+):.*loss=([\d\.]+).*RMSE_E_per_atom=\s+([\d\.]+).*RMSE_F=\s+([\d\.]+)"

    with open(log_file, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                rmse_e = float(match.group(3))
                rmse_f = float(match.group(4))

                epochs.append(epoch)
                loss_values.append(loss)
                rmse_e_values.append(rmse_e)
                rmse_f_values.append(rmse_f)

    # 创建保存图像的目录
    output_dir = "../experiment/plot"
    os.makedirs(output_dir, exist_ok=True)

    # 提取日志文件的名称（不包含路径和扩展名）
    log_filename = os.path.splitext(os.path.basename(log_file))[0]
    output_file = os.path.join(output_dir, f"{log_filename}.png")

    # 绘制折线图
    plt.figure(figsize=(10, 6))

    # 绘制 loss 曲线
    plt.subplot(3, 1, 1)
    plt.plot(epochs, loss_values, marker='o', color='b', label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.legend()

    # 绘制 RMSE_E 曲线
    plt.subplot(3, 1, 2)
    plt.plot(epochs, rmse_e_values, marker='o', color='g', label='RMSE_E')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE_E (meV)')
    plt.title('RMSE_E vs Epoch')
    plt.legend()

    # 绘制 RMSE_F 曲线
    plt.subplot(3, 1, 3)
    plt.plot(epochs, rmse_f_values, marker='o', color='r', label='RMSE_F')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE_F (meV / A)')
    plt.title('RMSE_F vs Epoch')
    plt.legend()

    plt.tight_layout()  # 调整布局

    # 保存图像到指定目录
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Plot loss and RMSE values from a log file.")
    parser.add_argument('log_file', type=str, help='Path to the log file')
    args = parser.parse_args()

    # 调用绘图函数
    plot_log_data(args.log_file)