
import numpy as np
import matplotlib
import random
import time
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

plt.rcParams['path.simplify'] = True
plt.rcParams['path.simplify_threshold'] = 0.1


def plot_activate(input_activate, image_name="model-activate.pdf"):
    print(f"plot input activate, save image name: {image_name}")

    # 创建行列索引网格
    rows, cols = input_activate.shape

    x = np.arange(0, cols)  # 列索引作为x轴
    y = np.arange(0, rows)  # 行索引作为y轴

    X, Y = np.meshgrid(x, y)  # 生成坐标网格

    # 创建三维图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    bottom = np.zeros_like(input_activate)

    colors_element=["red", "yellow", "blue", "green", "orange"]
    colors=[]
    temp = random.choices(colors_element, k=cols)
    for i in range(0, rows):
        colors.extend(temp)

    colors_np =np.array(colors)

    ax.bar3d(
        X.ravel(), Y.ravel(), bottom.ravel(), 0.05, 0.05, np.abs(input_activate).ravel(),
        color=colors_np,
        shade=True,
    )

    # 添加标签和标题
    ax.set_xlabel('channel', labelpad=15)
    ax.set_ylabel('token', labelpad=15)
    ax.set_zlabel('Value', labelpad=10)
    ax.set_title('Input Activate', pad=20)

    start = time.perf_counter()
    plt.savefig(image_name, dpi=100)
    end = time.perf_counter()
    exe_time = end - start
    print(f"执行时间：{exe_time:.6f} 秒")


