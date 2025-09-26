import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
from scipy import stats
# 绘制热图
def cal_heatmap(data: pd.DataFrame, output=None, row_height=None, cmap='viridis', cbar_label='set label', vmin=None, vmax=None, bound_ticks='both', log_scale_cbar=False) -> plt.Figure:
    """
    计算并绘制热图，支持对数映射的颜色条。

    Parameters
    ----------
    data : pd.DataFrame
        输入的数据框，行为样本，列为特征。
    output : str, optional
        如果提供，热图将保存为指定路径的图片文件。默认值为 None。
    row_height : float, optional
        设置每一行的高度。默认值为 None。
    cmap : str, optional
        颜色映射表。默认值为 'viridis'。
    cbar_label : str, optional
        颜色条的标签。默认值为 'set label'。
    vmin : float, optional
        颜色映射的最小值。默认值为 None。
    vmax : float, optional
        颜色映射的最大值。默认值为 None。
    log_scale_cbar : bool, optional
        控制是否应用对数颜色条映射。默认值为 False。
        
    Returns
    -------
    plt.Figure
        一个包含绘图对象的 Matplotlib Figure 对象，可用于进一步修改或保存图像。
    
    Examples
    --------
    >>> df = pd.DataFrame(np.random.rand(10, 5), columns=list('ABCDE'))
    >>> alpha_df = pd.DataFrame(np.random.rand(10, 5), columns=list('ABCDE'))
    >>> fig = cal_heatmap(df, output='heatmap', row_height=0.6, alpha_data=alpha_df, vmin=0, vmax=1)
    >>> fig.savefig('heatmap.png')
    """
    from matplotlib.colors import LogNorm
    
    # 计算图形大小
    if row_height is not None:
        num_rows = data.shape[1]
        figsize = (10, row_height * num_rows)
    else:
        figsize = (8, 14)

    # 创建图形和轴对象
    fig, ax = plt.subplots(figsize=figsize)
    
    # 确定是否使用对数映射，检查 vmin 是否大于 0
    if log_scale_cbar:
        if vmin is None or vmin <= 0:
            norm = LogNorm(vmin=1, vmax=vmax)
        else:
            norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = None
        
    # 使用 heatmap 绘制热图
    sns.heatmap(
        data,
        ax=ax,
        annot=True,
        fmt='.2f',
        cmap=cmap,
        linewidths=0.,
        linecolor='gray',
        cbar_kws={"shrink": 0.95, 'orientation':'horizontal', 
                  'label': cbar_label,
                  'pad':0.02,},
        vmin=vmin,
        vmax=vmax,
        norm=norm  # 应用对数或线性映射
    )

    # 设置列标签
    ax.set_xticklabels(list(data.columns), rotation=45)
    ax.set_xlabel('')
    # 将列标签放在上端
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    
    # 设置行标签
    ax.set_yticklabels([f'{col}' for col in data.index], rotation=0)
    # ax.set_yticklabels([f'{col} (N={len(gene_set_dict[col])})' for col in data.columns], rotation=0)
    ax.set_ylabel('')

    # 移除刻度线
    ax.tick_params(axis='both', which='both', length=0)
    
    # 获取 color bar 对象
    cbar = ax.collections[0].colorbar
    # cbar.set_label(r'$-log_{10}(P)$', fontsize=12)
    
    # 替换颜色条标签
    cbar_ticks = cbar.get_ticks()
    if bound_ticks=='both':
        cbar_labels = [f"$\geq {cbar_ticks[-1]:.2f}$" if tick == cbar_ticks[-1] else f"$\leq {cbar_ticks[0]:.2f}$" if tick == cbar_ticks[0] else f"${tick:.2f}$" for tick in cbar_ticks]
    elif bound_ticks=='lower':
        cbar_labels = [f"$\leq {cbar_ticks[0]:.2f}$" if tick == cbar_ticks[0] else f"${tick:.2f}$" for tick in cbar_ticks]
    elif bound_ticks=='upper':
        cbar_labels = [f"$\geq {cbar_ticks[-1]:.2f}$" if tick == cbar_ticks[-1] else f"${tick:.2f}$" for tick in cbar_ticks]
    else:
        cbar_labels = [tick for tick in cbar_ticks]

    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels(cbar_labels)
    
    # 保存图像文件，如果指定了输出路径
    if output is not None:
        # plt.tight_layout()
        fig.savefig(f'{output}.jpg', dpi=360, pad_inches=0.1, bbox_inches='tight')
        fig.savefig(f'{output}.svg', pad_inches=0.1, bbox_inches='tight')
        print(f'output: {output}.jpg')
        
    # 返回 Figure 对象
    return fig, ax



def plot_bubble_heatmap(size_matrix, label_matrix=None, color_matrix=None, output=None,
                        cmap='viridis', cbar_label='set label',
                        vmin=None, vmax=None, bound_ticks=None,
                        size_label = 'size legend',
                        marker_shape='o', show_grid=True, cbar_kws=None):
    """
    绘制一个带有可调形状的圆圈大小和颜色映射的热图。

    参数：
    - size_matrix: 用于映射圆圈大小的矩阵，可以是numpy数组或pandas DataFrame。
    - label_matrix: 用于显示在圆圈上的标签矩阵。如果未提供，则使用size_matrix的值。
    - color_matrix: 用于映射圆圈颜色的矩阵。如果未提供，则使用size_matrix的值。
    - cmap: 颜色映射，默认'viridis'。
    - cbar_label: 颜色条的标签。
    - vmin, vmax: 颜色映射的最小值和最大值。
    - bound_ticks: 控制颜色条标签的逻辑，可选值为'both', 'lower', 'upper'或其他。
    - marker_shape: 气泡的形状，例如'o'，'s'，'D'，'v'等。
    - show_grid: 是否显示背景网格，默认False。
    """
    from matplotlib.lines import Line2D
    from matplotlib.colors import Normalize
    
    # 检查输入是否为DataFrame，并提取行和列标签
    if isinstance(size_matrix, pd.DataFrame):
        row_labels = size_matrix.index.astype(str)
        col_labels = size_matrix.columns.astype(str)
    else:
        n, m = size_matrix.shape
        row_labels = np.arange(n).astype(str)
        col_labels = np.arange(m).astype(str)
        size_matrix = pd.DataFrame(size_matrix, index=row_labels, columns=col_labels)

    if label_matrix is None:
        label_matrix = size_matrix.copy()
    elif not isinstance(label_matrix, pd.DataFrame):
        label_matrix = pd.DataFrame(label_matrix, index=size_matrix.index, columns=size_matrix.columns)
    else:
        # 确保索引和列标签匹配
        label_matrix = label_matrix.reindex(index=size_matrix.index, columns=size_matrix.columns)

    if color_matrix is None:
        color_matrix = size_matrix.copy()
    elif not isinstance(color_matrix, pd.DataFrame):
        color_matrix = pd.DataFrame(color_matrix, index=size_matrix.index, columns=size_matrix.columns)
    else:
        # 确保索引和列标签匹配
        color_matrix = color_matrix.reindex(index=size_matrix.index, columns=size_matrix.columns)

    n, m = size_matrix.shape

    # 准备x和y的位置
    x = np.arange(m)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y)

    sizes = size_matrix.values.flatten()
    labels = label_matrix.values.flatten()
    colors = color_matrix.values.flatten()
    X = X.flatten()
    Y = Y.flatten()

    # 创建掩码，过滤掉sizes或colors中的NaN值
    mask = (~np.isnan(sizes)) & (~np.isnan(colors))
    sizes = sizes[mask]
    labels = labels[mask]
    colors = colors[mask]
    X = X[mask]
    Y = Y[mask]

    # 定义圆圈大小，根据标签长度自动调整最小大小
    # 计算标签所需的最小气泡大小
    label_lengths = np.array([len(str(label)) for label in labels])
    fontsize = 8  # 字体大小
    # 假设每个字符宽度为字体大小的一半，可以调整该估计
    min_bubble_size = np.max(label_lengths) * fontsize * 5

    min_size = min_bubble_size  # 最小圆圈大小
    max_size = 2000  # 最大圆圈大小

    # 使用np.nanmin和np.nanmax计算最小值和最大值
    size_min = np.nanmin(sizes)
    size_max = np.nanmax(sizes)

    if size_max == size_min:
        sizes_norm = np.ones_like(sizes)
    else:
        sizes_norm = (sizes - size_min) / (size_max - size_min)
    sizes_plot = sizes_norm * (max_size - min_size) + min_size

    # 设置颜色映射的最小值和最大值
    if vmin is None:
        vmin = np.nanmin(colors)
    if vmax is None:
        vmax = np.nanmax(colors)

    norm = Normalize(vmin=vmin, vmax=vmax)

    # 自动设置figsize
    figsize = (m * 0.8 + 2, n * 0.8 + 1)
    fig, ax = plt.subplots(figsize=figsize)

    # 主绘图区域
    scatter = ax.scatter(X, Y, s=sizes_plot, c=colors, cmap=cmap, norm=norm, marker=marker_shape)

    # 自动设置字体颜色
    cmap_instance = plt.get_cmap(cmap)
    color_values = cmap_instance(norm(colors))
    # 计算感知亮度
    luminance = (0.299 * color_values[:, 0] + 0.587 * color_values[:, 1] + 0.114 * color_values[:, 2])
    text_colors = ['white' if l < 0.5 else 'black' for l in luminance]

    # 添加文本标签
    for i in range(len(X)):
        ax.text(X[i], Y[i], f'{labels[i]}', ha='center', va='center', fontsize=fontsize, color=text_colors[i])

    # 设置坐标轴
    ax.set_xlim(-0.5, m - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xticks(np.arange(m))
    ax.set_yticks(np.arange(n))
    
    ax.xaxis.set_ticks_position('top') # 列标签置于上方
    ax.xaxis.set_label_position('top') 
    ax.set_xticklabels(col_labels, rotation=45, ha='left')  # 将对齐方式设置为左对齐
    
    ax.set_yticklabels(row_labels, rotation=0)
    ax.invert_yaxis()  # 反转y轴，使得第一个矩阵索引在顶部
    ax.set_aspect('equal')

    # 控制背景网格显示
    ax.grid(show_grid)
    # ax.set_facecolor('none')
    
    # ==== 添加颜色条 ====
    # 自定义颜色条的位置和宽度
    # 调整 [left, bottom, width, height]，值的范围是 0 到 1，表示相对于fig的比例
    cbar_left = 0.88  # 颜色条的左边界
    cbar_width = 0.04  # 颜色条的宽度
    cbar_bottom = 0.11  # 颜色条的下边界
    cbar_height = 0.45  # 颜色条的高度

    cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])

    cbar = plt.colorbar(scatter, cax=cbar_ax, **(cbar_kws or {}))
   
    # 可以根据需要调整标签与颜色条之间的间距
    # cbar.ax.xaxis.labelpad = 10  # 设置标签与颜色条之间的间距，默认值为10，增大值可以增大间距
    
    # 处理颜色条标签
    cbar_ticks = cbar.get_ticks()
    if bound_ticks == 'both':
        cbar_labels = [f"$\geq {cbar_ticks[-1]:.2f}$" if tick == cbar_ticks[-1]
                       else f"$\leq {cbar_ticks[0]:.2f}$" if tick == cbar_ticks[0]
                       else f"${tick:.2f}$" for tick in cbar_ticks]
    elif bound_ticks == 'lower':
        cbar_labels = [f"$\leq {cbar_ticks[0]:.2f}$" if tick == cbar_ticks[0]
                       else f"${tick:.2f}$" for tick in cbar_ticks]
    elif bound_ticks == 'upper':
        cbar_labels = [f"$\geq {cbar_ticks[-1]:.2f}$" if tick == cbar_ticks[-1]
                       else f"${tick:.2f}$" for tick in cbar_ticks]
    else:
        cbar_labels = [f"${tick:.2f}$" for tick in cbar_ticks]

    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels(cbar_labels)
    
    # 设置颜色条的标签（默认方法）
    # cbar.set_label(cbar_label)
    # ==== 手动添加颜色条标签 ====
    # 使用 cbar_ax.text() 方法手动在颜色条顶部添加标签
    cbar.ax.text(0.8, 1.05, cbar_label, ha='center', va='bottom', fontsize=10, transform=cbar_ax.transAxes)
    
    
    # ==== 添加大小图例 ====
    # 自定义大小图例的位置和尺寸
    # 调整 [left, bottom, width, height]
    legend_left = 0.875  # 大小图例的左边界
    legend_width = 0.11  # 大小图例的宽度
    legend_bottom = 0.58  # 大小图例的下边界
    legend_height = 0.33  # 大小图例的高度

    size_legend_ax = fig.add_axes([legend_left, legend_bottom, legend_width, legend_height])

    size_values = np.linspace(size_min, size_max, num=5)
    size_labels = [f'{val:.2f}' for val in size_values]
    if size_max == size_min:
        size_values_norm = np.ones_like(size_values)
    else:
        size_values_norm = (size_values - size_min) / (size_max - size_min)
    size_values_plot = size_values_norm * (max_size - min_size) + min_size

    size_legend_ax.axis('off')

    # 计算图例中每个标记的位置
    legend_y = np.linspace(0.2, 0.8, len(size_values))

    # ==== 调整标记和标签之间的间距 ====
    # 可以修改 scatter 和 text 的 x 坐标，例如，将 0.3 和 0.6 改为其他值

    for y_pos, s, label in zip(legend_y, size_values_plot[::-1], size_labels[::-1]):
        size_legend_ax.scatter([0.3], [y_pos], s=s, color='gray', marker=marker_shape)
        size_legend_ax.text(0.6, y_pos, label, va='center', fontsize=8)
        # 在这里调整图例标记和标签之间的间距
        # 例如，修改上面 scatter 的 x 坐标 0.3 和 text 的 x 坐标 0.6

    size_legend_ax.set_xlim(0, 1)
    size_legend_ax.set_ylim(0, 1)
    size_legend_ax.set_title(size_label, fontsize=10)
    size_legend_ax.title.set_position([0.5, 0.875])  # 进一步微调标题位置
    

    # 保存图像文件，如果指定了输出路径
    if output is not None:
        # plt.tight_layout()
        fig.savefig(f'{output}.jpg', dpi=360, pad_inches=0.1, bbox_inches='tight')
        fig.savefig(f'{output}.svg', pad_inches=0.1, bbox_inches='tight')
        fig.savefig(f'{output}.pdf', pad_inches=0.1, bbox_inches='tight')
        print(f'output: {output}.jpg')
        
    # 返回 Figure 对象
    return fig, ax

def create_custom_colormap(cmap_name='RdYlBu_r', vmin=0, vmax=100, set_mid=50):
    """
    创建一个自定义的渐变调色板，将特定值设置为中点。
    
    参数：
    cmap_name: str, 预设的调色板名称，如 'RdYlBu_r' 等。
    vmin: float, 调色板的最小值。
    vmax: float, 调色板的最大值。
    set_mid: float, 你希望设置为中点的值。
    
    返回值：
    一个自定义的 LinearSegmentedColormap 对象。
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    # 获取预设调色板
    base_cmap = plt.get_cmap(cmap_name)
    
    # 创建 TwoSlopeNorm 实例来设置自定义中点
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=set_mid, vmax=vmax)
    
    # 创建一个新函数来映射原始 cmap 到新的范围
    def colormap_map(x):
        # 计算归一化值
        norm_value = norm(x)
        # 获取 cmap 中的颜色
        return base_cmap(norm_value)
    
    # 创建一个新的颜色列表
    color_list = [colormap_map(i) for i in np.linspace(vmin, vmax, 256)]
    
    # 创建一个自定义的 LinearSegmentedColormap
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(f'{cmap_name}_custom', color_list)
    
    return custom_cmap