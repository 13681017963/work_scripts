import matplotlib.pyplot as plt


def parameters_map50(algorithms, parameters_number, map50, labels):
    for idx, algorithm in enumerate(algorithms):
        # 绘制图形
        plt.plot(parameters_number[idx], map50[idx], label=algorithm, marker='o')
        for i, label in enumerate(labels[idx]):
            if i == 0:
                plt.text(parameters_number[idx][i], map50[idx][i], label, ha='left', va='bottom')
            else:
                plt.text(parameters_number[idx][i], map50[idx][i], label, ha='center', va='bottom')
    plt.xlabel('Number of Parameters (M)')
    plt.ylabel('$mAP_{val}^{50}$')
    plt.title('Number of Parameters Comparison of YOLOv8 and YOLOv4')
    plt.legend(loc='lower right')
    plt.show()


def speed_map50(algorithms, parameters_number, map50, labels):
    for idx, algorithm in enumerate(algorithms):
        # 绘制图形
        plt.plot(parameters_number[idx], map50[idx], label=algorithm, marker='o')
        for i, label in enumerate(labels[idx]):
            if i == 0:
                plt.text(parameters_number[idx][i], map50[idx][i], label, ha='left', va='bottom')
            else:
                plt.text(parameters_number[idx][i], map50[idx][i], label, ha='center', va='bottom')
    plt.xlabel('Latency 2080ti TensorRT FP16(ms/img)')
    plt.ylabel('$mAP_{val}^{50}$')
    plt.title('Inference Speed Comparison of YOLOv8 and YOLOv4')
    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    # 模拟数据
    algorithms = ['YOLOv8', 'YOLOv4']
    parameters_number = [[3.0, 11.1, 25.8, 43.6, 68.2], [3.0, 15.0, 25.8, 43.6, 68.2]]  # YOLO网络规模，单位是M
    map50 = [[0.76, 0.79, 0.82, 0.85, 0.87], [0.72, 0.90508, 0.78, 0.81, 0.83]]
    labels = [['n', 's', 'm', 'l', 'x'], ['mobilenet_v2', 'mobilenet_v1', 'c', 'd', 'e']]
    parameters_map50(algorithms, parameters_number, map50, labels)

    # 模拟数据
    speed = [[3.0, 11.1, 25.8, 43.6, 68.2], [3.0, 15.0, 25.8, 43.6, 68.2]]  # YOLO网络规模，单位是M
    speed_map50(algorithms, speed, map50, labels)
