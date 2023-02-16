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
    plt.ylabel('$mAP_{val}^{50-95}$')
    plt.title('Number of Parameters Comparison of YOLOv8 and TAO YOLOv4')
    plt.legend(loc='lower right')
    # 添加水平箭头
    plt.annotate("faster", xy=(20, 0.72), xytext=(50, 0.72125),
                 arrowprops=dict(facecolor='black', shrink=100.0),
                 ha='left')
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
    plt.ylabel('$mAP_{val}^{50-95}$')
    plt.title('Inference Speed Comparison of YOLOv8 and TAO YOLOv4')
    plt.legend(loc='lower right')
    # 添加水平箭头
    plt.annotate("faster", xy=(20, 0.75), xytext=(50, 0.75125),
                arrowprops=dict(facecolor='black', shrink=100.0),
                ha='left')
    plt.show()


if __name__ == '__main__':
    # 模拟数据
    algorithms = ['YOLOv8', 'TAO YOLOv4 unpruned', 'TAO YOLOv4 pruned']
    parameters_number = [[3.0, 11.1, 25.8, 43.6, 68.2], [15.0, 24.9, 34.8, 57.0, 85.3], [8.9, 15.0, 17.4, 23.3, 30.0]]  # YOLO网络规模，单位是M
    map50_95 = [[0.72, 0.753, 0.777, 0.785, 0.792], [0.72, 0.75, 0.78, 0.81, 0.83], [0.72, 0.75, 0.78, 0.81, 0.83]]
    labels = [['n', 's', 'm', 'l', 'x'], ['mobilenet_v2', 'mobilenet_v1', 'resnet18', 'darknet19', 'resnet50'], ['mobilenet_v2', 'mobilenet_v1', 'resnet18', 'darknet19', 'resnet50']]
    parameters_map50(algorithms, parameters_number, map50_95, labels)

    # 模拟数据
    # speed = [[3.0, 11.1, 25.8, 43.6, 68.2], [3.0, 15.0, 25.8, 43.6, 68.2]]  # YOLO网络规模，单位是M
    # speed_map50(algorithms, speed, map50_95, labels)
