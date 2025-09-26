# %%

'''
使用SEM解释临床特征-潜变量-聚类-临床结局之间的关系
'''
import os, sys
import re
from datetime import datetime
import argparse
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
import networkx as nx
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path
import matplotlib.patches as mpatches
from matplotlib import cm, colors
import matplotlib.lines as mlines
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

pd.set_option('display.max_rows', 300)
pd.set_option('display.max_columns', 1000)
sns.set_theme(style="whitegrid")
# 检测运行环境
IN_NOTEBOOK = None
def in_notebook():
    return 'IPKernelApp' in getattr(globals().get('get_ipython', lambda: None)(), 'config', {})
    
if in_notebook():
    IN_NOTEBOOK = True
    print(f'run in notebook')
    from IPython.display import clear_output, display
    pd.set_option('display.max_columns', None)
    notebook_dir = os.getcwd()
    src_path = os.path.abspath(os.path.join(notebook_dir, '..'))
    TARGET = 'mortality'
    DATASET_NAME = 'EXIT_SEP'
else:
    IN_NOTEBOOK = False
    src_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-target', metavar='mortality', type=str, default='mortality',help='''mortality: 28d和院内死亡建模; shock: 感染性休克建模''')
    parser.add_argument('-dataset', metavar= 50, type=str, default='MIMIC_IV',help='''建模数据集 [MIMIC_IV, eICU, EXIT_SEP]''')
    sys_args = parser.parse_args()
    TARGET = sys_args.target
    DATASET_NAME = sys_args.dataset
    print(f'run in script')

sys.path.append(src_path) if src_path not in sys.path else None
from src.utils import *
from src.model_utils import *
from src.metrix import cal_ci, format_ci, bootstrap_resampler
from src.setup import *
from risk_setup import *
from SUAVE import SuaveClassifier
import semopy
from semopy import Model, calc_stats

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'current device: {DEVICE}')
print(f'current dataset: {DATASET_NAME}')
# existing
risk_model_path = f'{ANALYSIS}/{TARGET}_risk_models/'
cluster_path = f'{risk_model_path}/clustering/{DATASET_NAME}/'
model_explain_path = f'{risk_model_path}/model_explain/{DATASET_NAME}/'

# %%
df_original_feature = pd.read_csv(f'{cluster_path}/{DATASET_NAME}_clusterred-inverse_missing.tsv.gz', sep='\t', index_col='ID')
df = pd.read_csv(f'{DATA}/imputed/{DATASET_NAME}_SEM_imputed.tsv.gz', sep='\t', index_col='ID')

latent_cols = list(df.columns[df.columns.str.contains('latent_feature')])
SEM_CLINICAL_FEATURES, cont_vars, cate_vars, outcomes, cont_features, cate_features = get_sem_vars(DATASET_NAME)
cluster_cols = list(df.columns[df.columns.str.contains('cluster')])
clinical_features = [f for f in SEM_CLINICAL_FEATURES.keys() if not f in outcomes]
exclude_clinical_feature = ['CRRT']
picked_clinical_features = [f for f in clinical_features if not f in exclude_clinical_feature]

# 选择 cluster_4 作为参照 可以提升SEM拟合度 但可视化解释牺牲了一个类别
# 我们使用 dummy coding，将 cluster_4 作为 baseline，故模型中仅纳入了 cluster_1, cluster_2, cluster_3 三个哑变量，以避免共线性。
# 当 cluster_1, cluster_2, cluster_3 均为 0 时，即表示 cluster_4 的效应
# cluster_3 is the reference category (not shown in the diagram).

# df.drop(columns=cluster_cols[3], inplace=True)
# cluster_cols = list(df.columns[df.columns.str.contains('cluster')])

# %%
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                       for i in range(X.shape[1])]
    return vif_data

vif_df = calculate_vif(df[clinical_features])

# 移除常数项（VIF = 1 无需关注）
vif_df = vif_df[vif_df['feature'] != 'const']

print(vif_df)

# %% [markdown]
# # 构建SEM

# %%
# 0) clinical_features -> clinical_features
#   # 根据医学知识定义临床变量间的路径
clinical_block = [
                #   ('PaO2_FiO2_ratio ~ Respiratory_Support'),
                #   ('PaCO2 ~ Respiratory_Support'),
                  ('PaO2 ~ Respiratory_Support'),

                  ('WBC ~ NE_count + LYM_count'),
                #   ('STB ~ ALT + AST'),
                  
                #   ('Lac ~ CRRT'),
                #   ('PH ~ CRRT'),
                #   ('HCO3_ ~ CRRT'),
                #   ('K_ ~ CRRT'),
                #   ('Na_ ~ CRRT'),
                #   ('BUN ~ CRRT'),
                #   ('Scr ~ CRRT'),
                  ]

# 1) clinical_features -> latent_features
#   生成形如:
#   latent_feature_1 ~ age + sex + temperature + ...
#   latent_feature_2 ~ age + sex + temperature + ...
latent_block = []
for lv in latent_cols:
    rhs = " + ".join(picked_clinical_features)  # 把所有临床特征加起来
    latent_block.append(f"{lv} ~ {rhs}")
    
# 2) latent_features / clinical_features -> clusters
#   生成形如:
#   cluster_1 ~ latent_feature_1 + latent_feature_2 + ...
#   cluster_2 ~ latent_feature_1 + latent_feature_2 + ...
cluster_block = []
rhs_latent = " + ".join(latent_cols)
for c in cluster_cols:
    cluster_block.append(f"{c} ~ {rhs_latent}")

cluster_block2 = []
rhs_clinical = " + ".join(picked_clinical_features)
for c in cluster_cols:
    cluster_block2.append(f"{c} ~ {rhs_clinical}")
    
# 3) clusters -> outcomes
#   生成形如:
#   in_hospital_mortality ~ cluster_1 + cluster_2
#   28d_mortality ~ cluster_1 + cluster_2
outcome_block = []
rhs_clusters = " + ".join(cluster_cols)
for o in outcomes:
    outcome_block.append(f"{o} ~ {rhs_clusters}")
    
# 让 latent_features 和cluster 一起直接作用于 outcome:
outcome_block_2 = []
rhs_full = " + ".join(latent_cols + cluster_cols)
for o in outcomes:
    outcome_block_2.append(f"{o} ~ {rhs_full}")
    
# 纯临床变量推导 outcome:
outcome_block_3 = []
rhs_clinical = " + ".join(clinical_features)
for o in outcomes:
    outcome_block_3.append(f"{o} ~ {rhs_clinical}")
    
# 纯潜变量推导 outcome:
outcome_block_5 = []
rhs_latent = " + ".join(latent_cols)
for o in outcomes:
    outcome_block_5.append(f"{o} ~ {rhs_latent}")
    

# %%
SEM_RESULTS_DICT = {}
# 构建 SEM 模型: clinical -> latent -> cluster -> outcomes
print(f'主分析: clinical -> latent -> cluster -> outcomes')
model_desc = "\n".join(clinical_block + latent_block + cluster_block + outcome_block)
model = Model(model_desc)
model.fit(df)
main_results = model.inspect() # 3. 查看结果
main_stats = calc_stats(model) # 4. 拟合度指标
show_dataframe(main_stats)
SEM_RESULTS_DICT['primary'] = main_results

with pd.ExcelWriter(f'{model_explain_path}/SEM_outcomes_cluster-.xlsx') as writer:
    main_results.to_excel(writer, index=False, sheet_name='SEM model')
    main_stats.to_excel(writer, index=False, sheet_name='model fitness')

# 构建 SEM 模型2: latent 和 cluster 一起作用于 outcome (分析latent variables 是否有额外解释作用)
print(f'SEM 2 : latent 和 cluster 一起作用于 outcome')
model_desc_2 = "\n".join(clinical_block + latent_block + cluster_block + outcome_block_2)
model = Model(model_desc_2)
model.fit(df)
results = model.inspect() # 3. 查看结果
stats = calc_stats(model) # 4. 拟合度指标
show_dataframe(stats)
SEM_RESULTS_DICT[2] = results

with pd.ExcelWriter(f'{model_explain_path}/SEM2_outcomes_latent+cluster.xlsx') as writer:
    results.to_excel(writer, index=False, sheet_name='SEM model')
    stats.to_excel(writer, index=False, sheet_name='model fitness')
    
# 构建 SEM 模型3: 纯临床变量推导 (考虑临床变量间因果) outcome
print(f'SEM 3 : 纯临床变量推导 (考虑临床变量间因果) outcome')
model_desc_3 = "\n".join(clinical_block + outcome_block_3)
model = Model(model_desc_3)
model.fit(df)
results = model.inspect() # 3. 查看结果
stats = calc_stats(model) # 4. 拟合度指标
show_dataframe(stats)
SEM_RESULTS_DICT[3] = results

with pd.ExcelWriter(f'{model_explain_path}/SEM3_outcomes_clinical.xlsx') as writer:
    results.to_excel(writer, index=False, sheet_name='SEM model')
    stats.to_excel(writer, index=False, sheet_name='model fitness')
    

# 构建 SEM 模型4: 纯临床变量推导 (不考虑临床变量间因果) outcome
print(f'SEM 4: 纯临床变量推导 (假设临床变量间独立) outcome')
model_desc_4 = "\n".join(outcome_block_3)
model = Model(model_desc_4)
model.fit(df)
results = model.inspect() # 3. 查看结果
stats = calc_stats(model) # 4. 拟合度指标
show_dataframe(stats)
SEM_RESULTS_DICT[4] = results

with pd.ExcelWriter(f'{model_explain_path}/SEM4_outcomes_clinical-independent.xlsx') as writer:
    results.to_excel(writer, index=False, sheet_name='SEM model')
    stats.to_excel(writer, index=False, sheet_name='model fitness')
    

# 构建 SEM 模型5: 纯潜变量推导 outcome
print(f'SEM 5: 纯潜变量推导 outcome')
model_desc_5 = "\n".join(outcome_block_5)
model = Model(model_desc_5)
model.fit(df)
results = model.inspect() # 3. 查看结果
stats = calc_stats(model) # 4. 拟合度指标
show_dataframe(stats)
SEM_RESULTS_DICT[5] = results

with pd.ExcelWriter(f'{model_explain_path}/SEM5_outcomes_latent.xlsx') as writer:
    results.to_excel(writer, index=False, sheet_name='SEM model')
    stats.to_excel(writer, index=False, sheet_name='model fitness')
    

# 构建 SEM 模型6: clinical -> cluster -> outcome
print(f'SEM 6: 临床变量 + 表型推导 outcome')
model_desc_6 = "\n".join(cluster_block2 + outcome_block)
model = Model(model_desc_6)
model.fit(df)
results = model.inspect() # 3. 查看结果
stats = calc_stats(model) # 4. 拟合度指标
show_dataframe(stats)
SEM_RESULTS_DICT[6] = results

with pd.ExcelWriter(f'{model_explain_path}/SEM6_outcomes_cluster+clinical-independent.xlsx') as writer:
    results.to_excel(writer, index=False, sheet_name='SEM model')
    stats.to_excel(writer, index=False, sheet_name='model fitness')
    
# 构建 SEM 模型7: clinical -> cluster -> outcome
print(f'SEM 7: 临床变量(考虑内部相关) + 表型推导 outcome')
model_desc_7 = "\n".join(clinical_block + cluster_block2 + outcome_block)
model = Model(model_desc_7)
model.fit(df)
results = model.inspect() # 3. 查看结果
stats = calc_stats(model) # 4. 拟合度指标
show_dataframe(stats)
SEM_RESULTS_DICT[7] = results

with pd.ExcelWriter(f'{model_explain_path}/SEM7_outcomes_cluster+clinical.xlsx') as writer:
    results.to_excel(writer, index=False, sheet_name='SEM model')
    stats.to_excel(writer, index=False, sheet_name='model fitness')

# %% [markdown]
# # 可视化 SEM

# %%
# 计算节点坐标
def assign_positions(x_layer, y_offset, node_list):
    """
    计算节点的坐标位置。

    给定 x 坐标、y 间距和节点列表，返回一个字典:
    {
      节点名称: (x, y)
    }
    其中最中间的节点位于 y=0，其他节点按照 y_offset 向上或向下均匀排布。

    参数
    ----------
    x_layer : float
        该层网络在 x 轴的位置（固定值）。
    y_offset : float
        相邻节点在 y 轴的间隔。
    node_list : list
        要排列的节点名称列表，顺序即是想要从上到下或下到上的顺序。

    返回
    ----------
    pos_dict : dict
        形如 {节点名称: (x, y)} 的坐标映射。
    """
    n = len(node_list)
    # 若只有 1 个节点，就直接放在 y=0
    if n == 1:
        return {node_list[0]: (x_layer, 0)}

    # 计算“中间下标”的浮点数形式，以便让中间节点落在 y=0
    # 例如，若 n=3，则 mid=(3-1)/2=1.0 => 索引为 [0,1,2] 分别映射到 y= -y_offset, 0, +y_offset
    # 若 n=4，则 mid=1.5 => [0,1,2,3] 分别映射到 y= -1.5*y_offset, -0.5*y_offset, 0.5*y_offset, 1.5*y_offset
    mid = (n - 1) / 2

    pos_dict = {}
    for i, node in enumerate(node_list):
        # 让第 i 个节点相对于中间下标 mid 偏移
        y_coord = (mid - i) * y_offset
        pos_dict[node] = (x_layer, y_coord)

    return pos_dict

def draw_smooth_edge(ax, start, end, color, linewidth=2.0,
                     curv_control_1=1.0, curv_control_2=1.0, alpha=0.9, mutation_scale=10):
    """
    绘制平滑曲线箭头边。

    参数
    ----------
    ax : matplotlib.axes.Axes
        绘图的轴对象。
    start : tuple
        边的起点坐标 (x, y)。
    end : tuple
        边的终点坐标 (x, y)。
    color : str or tuple
        边的颜色。
    linewidth : float, optional
        边的线宽，默认为2.0。
    curv_control_1 : float, optional
        贝塞尔曲线的第一个控制点偏移量，默认为1.0。
    curv_control_2 : float, optional
        贝塞尔曲线的第二个控制点偏移量，默认为1.0。
    alpha : float, optional
        边的透明度，默认为0.9。
    mutation_scale : float, optional
        箭头大小的缩放因子，默认为10。
    """
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.path import Path
    # 计算两侧的贝塞尔控制点，让边呈曲线流动
    control1 = (start[0] + curv_control_1, start[1])
    control2 = (end[0] - curv_control_2, end[1])
    verts = [start, control1, control2, end]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    path = Path(verts, codes)

    patch = FancyArrowPatch(
        path=path, color=color,
        linewidth=linewidth, alpha=alpha,
        arrowstyle='->',
        mutation_scale=mutation_scale  # 控制箭头大小
    )
    ax.add_patch(patch)

def create_custom_colormap(cmap_name='RdYlBu_r', vmin=0, vmax=100, set_mid=50):
    """
    创建一个自定义的渐变调色板，将特定值设置为中点。

    参数
    ----------
    cmap_name : str, optional
        预设的调色板名称，如 'RdYlBu_r' 等，默认为 'RdYlBu_r'。
    vmin : float, optional
        调色板的最小值，默认为0。
    vmax : float, optional
        调色板的最大值，默认为100。
    set_mid : float, optional
        希望设置为中点的值，默认为50。

    返回
    ----------
    custom_cmap : matplotlib.colors.LinearSegmentedColormap
        自定义的线性分段颜色映射对象。
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

def visualize_sem_with_colormap(
    G, 
    pos, 
    cmap="coolwarm", 
    zero_center_color_and_weight=True,
    node_type_to_color=None,
    label_dict=None,
    line_weight=10,
    node_size_unit=None,
    clip_outlier_percentile=None,
    mutation_scale=10,
):
    """
    可视化带有权重的有向图，并根据权重使用颜色映射。

    参数
    ----------
    G : networkx.DiGraph
        带有 'weight' 属性的有向图。
    pos : dict
        节点坐标字典，格式为 {节点名称: (x, y)}。
    cmap : str, optional
        用于边颜色映射的matplotlib色图，默认为"coolwarm"。
    zero_center_color_and_weight : bool, optional
        是否将色图的中点设为0，默认为True。
    node_type_to_color : dict, optional
        节点类型到颜色的映射字典，格式为 {node_type: color}，默认为None。
    label_dict : dict, optional
        将节点名手动映射为标签的字典，格式为 {node_name: label}，默认为None。
    line_weight : float, optional
        线宽缩放因子，默认为10。
    node_size_unit : float, optional
        控制标签放置位置用的尺度因子，默认为None（默认值为0.05）。
    clip_outlier_percentile : tuple或None, optional
        形如 (l, u)，表示对边权重在第 l% 和 u% 分位数处做裁剪，默认为None。
    mutation_scale : float, optional
        控制箭头大小的缩放因子，默认为10。

    返回
    ----------
    fig : matplotlib.figure.Figure
        绘图的Figure对象。
    ax : matplotlib.axes.Axes
        绘图的Axes对象。
    """
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib import cm, colors
    import matplotlib.lines as mlines
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    font_size = 10
    fig, ax = plt.subplots(figsize=(12, 8))

    if node_size_unit is None:
        node_size_unit = 0.05

    # 1) 收集所有边的 weight
    edges_data = list(G.edges(data=True))
    if not edges_data:
        print("Graph has no edges, nothing to draw.")
        return fig, ax

    weights = np.array([data["weight"] for _, _, data in edges_data])

    # 2) 如果指定了 clip_outlier_percentile，则对 weights 做分位数裁剪
    if clip_outlier_percentile is not None:
        l, u = clip_outlier_percentile
        # 防止输入值非法，简单检查
        if not (0 <= l < u <= 100):
            raise ValueError(f"clip_outlier_percentile must be between 0 and 100, got {clip_outlier_percentile}.")
        lower_bound = np.percentile(weights, l)
        upper_bound = np.percentile(weights, u)
        # 裁剪
        weights_for_color = np.clip(weights, lower_bound, upper_bound)
    else:
        # 不裁剪
        weights_for_color = weights

    # 3) 设定 colormap 的归一化范围
    wmin, wmax = weights_for_color.min(), weights_for_color.max()
    # 若全部值相同，避免除以0
    if np.isclose(wmin, wmax):
        wmin -= 1e-9
        wmax += 1e-9

    if zero_center_color_and_weight:
        norm = TwoSlopeNorm(vmin=wmin, vcenter=0, vmax=wmax)
    else:
        norm = colors.Normalize(vmin=wmin, vmax=wmax)
    scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
    
    # 4) 绘制节点
    if node_type_to_color is None:
        node_type_to_color = {}

    node_by_type = {}
    for n, d in G.nodes(data=True):
        t = d.get("node_type", "Unknown")
        node_by_type.setdefault(t, []).append(n)

    for t, nodelist in node_by_type.items():
        color = node_type_to_color.get(t, "lightblue")
        nx.draw_networkx_nodes(
            G, pos=pos, nodelist=nodelist, node_color=color,
            node_size=300, ax=ax, label=t
        )

    # 5) 绘制节点标签
    if label_dict is None:
        label_dict = {}
    # 将第一层、其他层标签区分处理
    pos_label = {}
    font_size = 10
    for node, (px, py) in pos.items():
        layer = G.nodes[node].get('layer', 1)
        display_name = label_dict.get(node, node)
        if layer == 1:
            # 在节点左侧
            pos_label[node] = (px - node_size_unit*1.01, py)
            ha = 'right'
            va = 'center'
        else:
            # 在节点下方
            pos_label[node] = (px, py - node_size_unit*0.7)
            ha = 'center'
            va = 'top'
        ax.text(
            pos_label[node][0], pos_label[node][1],
            display_name,
            ha=ha, va=va,
            fontsize=font_size
        )

    # 6) 绘制平滑边
    for i, (u, v, data) in enumerate(edges_data):
        w = data["weight"]
        # 使用裁剪后的值进行颜色映射
        if clip_outlier_percentile is not None:
            # 同步 clip
            w_for_color = np.clip(w, lower_bound, upper_bound)
        else:
            w_for_color = w

        edge_color = scalar_map.to_rgba(w_for_color)
        
        # 根据映射颜色的 w 将粗细同步归一化到 [0.1, 1.1]
        abs_wmax = max(abs(wmax), abs(wmin))
        abs_wmin = 0
        w_for_width =  ((abs(w_for_color) - abs_wmin) / (abs_wmax - abs_wmin)) + 0.1 # 防止weight为 0
        lw = w_for_width * line_weight

        start = pos[u]
        end = pos[v]
        start_x, _ = start
        end_x, _ = end
        # 简单的左右弧度
        if start_x != end_x:
            curv = 0.75 * abs(start_x - end_x)
            curv_control_1 = curv
            curv_control_2 = curv
        else:
            curv_control_1 = -node_size_unit
            curv_control_2 = node_size_unit

        draw_smooth_edge(ax, start, end, color=edge_color, linewidth=lw,
                         curv_control_1=curv_control_1,
                         curv_control_2=curv_control_2,
                         alpha = max((abs(w_for_width) - 1e-3), 0.1) - 0.1, # w_for_width 已归一化到[0.1, 1.1], 若原始w < 1e-3, 则alpha为0
                        #  alpha = 0.7,
                         mutation_scale=mutation_scale,
                         )

    # 7) 添加 colorbar
    cbar = plt.colorbar(scalar_map, ax=ax, fraction=0.04, pad=0.00, orientation='horizontal')
    cbar.set_label("Standardized Path Coefficient", rotation=0, labelpad=10, fontsize=font_size)
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.xaxis.set_label_coords(0.5, 1.5)

    # 8) 添加节点类型图例
    legend_handles = []
    for t, color in node_type_to_color.items():
        # 创建圆形图例
        circle = mlines.Line2D([], [], color=color, marker='o', linestyle='None',
                            markersize=10, label=t)
        legend_handles.append(circle)
        # patch = mpatches.Patch(color=color, label=t)
        # legend_handles.append(patch)
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", # title='Node types',
                  ncol=3, fontsize=font_size)

    # 获取数据范围
    x_min, x_max = ax.get_xlim()

    # 根据需求扩展左右空间
    x_margin_left = 0.10 * (x_max - x_min)  # 左侧扩展 
    x_margin_right = 0.01 * (x_max - x_min)  # 右侧扩展 

    # 重新设置 x 轴范围
    ax.set_xlim(x_min - x_margin_left, x_max + x_margin_right)
    
    # ax.margins(x=0.18, y=0.05)
    plt.axis("off")
    plt.tight_layout()

    return fig, ax


# %% [markdown]
# # clinical -> latent -> cluster -> outcome

# %%
# 只绘制 P < 0.05 的关系
for key in ['primary', 2]:
    viz_results = SEM_RESULTS_DICT[key].copy()
    mask_sig = viz_results['p-value'] < 0.05
    viz_results = viz_results[mask_sig]

    G = nx.DiGraph()

    node_cit = {
        **{n:{'node_type': feature_type, 'layer': 1} for n, (data_type, feature_type) in SEM_CLINICAL_FEATURES.items() if (not n in outcomes) and (n in picked_clinical_features)}, # 临床特征
        **{n:{'node_type': 'Latent Variable', 'layer': 2} for n in latent_cols},  # Latent
        **{n:{'node_type': 'Phenotype', 'layer': 3} for n in cluster_cols}, # Cluster
        **{n:{'node_type': 'Outcome', 'layer': 4} for n in outcomes}, # Outcomes
        }

    edges = [(row['rval'], row['lval'], row['Estimate']) for _, row in viz_results.iterrows()]
    G.add_nodes_from(node_cit.items())
    G.add_weighted_edges_from(edges)
    G.remove_edges_from(nx.selfloop_edges(G))

    x_layer1, y_offset_layer1 = 0, 0.05 # clinical feature 层
    x_layer2, y_offset_layer2 = x_layer1 + 0.75, y_offset_layer1 * 2 # latent_feature 层
    x_layer3, y_offset_layer3 = x_layer2 + 0.50, y_offset_layer1 * 3 # cluster 层
    x_layer4, y_offset_layer4 = x_layer3 + 0.25, y_offset_layer1 * 10 # outcome 层

    pos = {
        **assign_positions(x_layer1, y_offset_layer1, picked_clinical_features), # clinical feature 层坐标
        **assign_positions(x_layer2, y_offset_layer2, latent_cols), # latent_feature 层坐标
        **assign_positions(x_layer3, y_offset_layer3, cluster_cols), # cluster 层坐标
        **assign_positions(x_layer4, y_offset_layer4, outcomes), # outcomes 层坐标
    }
    pos_x = [x for n, (x, y), in pos.items()]
    pos_y = [y for n, (x, y), in pos.items()]

    node_type_to_color = {
        "Demographic":   "#A8D5BA",
        "Vital":         "#F8B8B8",
        # "Acid-base":     "#D4B5E8",
        "Metabolic":     "#D4B5E8",
        "Renal":         "#A12D45",
        "Respiratory":   "#B3D9E8",
        "Immune":        "#FDD9A5",
        "Hematology":    "#E7C5A5",
        "Coagulative":   "#F7A072",
        "Hepatic":       "#6B4C4A",
        # "Other":         "#8E736C",
        "Latent Variable":"#1C3F58",
        "Phenotype":     "#6B5B95",
        "Outcome":       "#FF6F61",
    }

    label_dict = {
        **{n: re.sub('latent_feature_(.+)', r'$z_{\1}$', n) for n in latent_cols},
        **{n: re.sub('.+cluster_', 'Phenotype ', n) for n in cluster_cols},
        'sex':'Sex',
        'age': 'Age',
        'temperature': 'Temperature', 
        'heart_rate': 'Heart Rate', 
        'respir_rate': 'Respiratory Rate', 
        'Respiratory_Support': 'Respiratory Support', 
        'PaO2_FiO2_ratio': r'PaO$_{2}$/FiO$_{2}$', 
        'PaCO2': r'PaCO$_{2}$', 
        'NE_count': 'NE#', 
        'LYM_count': 'LYM#', 
        'HCO3_': r'HCO$_{3}^{-}$', 
        'K_': r'K$^{+}$', 
        'Na_': r'Na$^{+}$', 
        'in_hospital_mortality': 'In-Hospital Mortality',
        '28d_mortality': '28-Day Mortality',
    }

    
    fig, ax = visualize_sem_with_colormap(G=G, 
                                        pos=pos,
                                        cmap=plt.cm.RdBu_r,
                                        zero_center_color_and_weight=True,
                                        node_type_to_color=node_type_to_color,
                                        label_dict=label_dict, 
                                        line_weight=3,
                                        node_size_unit=y_offset_layer1,
                                        clip_outlier_percentile=(0., 98.5),
                                        mutation_scale=1,
                                        )
    for fmt in ['jpg', 'svg', 'pdf']:
        fig.savefig(f'{model_explain_path}/SME_explain_{key}.{fmt}', dpi=360, bbox_inches='tight')

# %% [markdown]
# # clinical  -> cluster -> outcome

# %%
# 只绘制 P < 0.05 的关系
for key in [6, 7]:
    viz_results = SEM_RESULTS_DICT[key].copy()
    mask_sig = viz_results['p-value'] < 0.05
    viz_results = viz_results[mask_sig]

    G = nx.DiGraph()

    node_cit = {
        **{n:{'node_type': feature_type, 'layer': 1} for n, (data_type, feature_type) in SEM_CLINICAL_FEATURES.items() if (not n in outcomes) and (n in picked_clinical_features)}, # 临床特征
        **{n:{'node_type': 'Phenotype', 'layer': 2} for n in cluster_cols}, # Cluster
        **{n:{'node_type': 'Outcome', 'layer': 3} for n in outcomes}, # Outcomes
        }

    edges = [(row['rval'], row['lval'], row['Estimate']) for _, row in viz_results.iterrows()]
    G.add_nodes_from(node_cit.items())
    G.add_weighted_edges_from(edges)
    G.remove_edges_from(nx.selfloop_edges(G))

    x_layer1, y_offset_layer1 = 0, 0.05 # clinical feature 层
    x_layer2, y_offset_layer2 = x_layer1 + 0.75, y_offset_layer1 * 6 # cluster 层
    x_layer3, y_offset_layer3 = x_layer2 + 0.50, y_offset_layer1 * 6 # outcome 层

    pos = {
        **assign_positions(x_layer1, y_offset_layer1, picked_clinical_features), # clinical feature 层坐标
        **assign_positions(x_layer2, y_offset_layer2, cluster_cols), # cluster 层坐标
        **assign_positions(x_layer3, y_offset_layer3, outcomes), # outcomes 层坐标
    }
    pos_x = [x for n, (x, y), in pos.items()]
    pos_y = [y for n, (x, y), in pos.items()]

    node_type_to_color = {
        "Demographic":   "#A8D5BA",
        "Vital":         "#F8B8B8",
        # "Acid-base":     "#D4B5E8",
        "Metabolic":     "#D4B5E8",
        "Renal":         "#A12D45",
        "Respiratory":   "#B3D9E8",
        "Immune":        "#FDD9A5",
        "Hematology":    "#E7C5A5",
        "Coagulative":   "#F7A072",
        "Hepatic":       "#6B4C4A",
        # "Other":         "#8E736C",
        # "Latent Variable":"#1C3F58",
        "Phenotype":     "#6B5B95",
        "Outcome":       "#FF6F61",
    }

    label_dict = {
        **{n: re.sub('latent_feature_(.+)', r'$z_{\1}$', n) for n in latent_cols},
        **{n: re.sub('.+cluster_', 'Phenotype ', n) for n in cluster_cols},
        'sex':'Sex',
        'age': 'Age',
        'temperature': 'Temperature', 
        'heart_rate': 'Heart Rate', 
        'respir_rate': 'Respiratory Rate', 
        'Respiratory_Support': 'Respiratory Support', 
        'PaO2_FiO2_ratio': r'PaO$_{2}$/FiO$_{2}$', 
        'PaCO2': r'PaO$_{2}$', 
        'PaCO2': r'PaCO$_{2}$', 
        'NE_count': 'NE#', 
        'LYM_count': 'LYM#', 
        'HCO3_': r'HCO$_{3}^{-}$', 
        'K_': r'K$^{+}$', 
        'Na_': r'Na$^{+}$', 
        'in_hospital_mortality': 'In-Hospital Mortality',
        '28d_mortality': '28-Day Mortality',
    }

    fig, ax = visualize_sem_with_colormap(G=G, 
                                        pos=pos,
                                        cmap=plt.cm.RdBu_r,
                                        node_type_to_color=node_type_to_color,
                                        label_dict=label_dict, 
                                        line_weight=3,
                                        node_size_unit=y_offset_layer1,
                                        clip_outlier_percentile=(0, 97.5)
                                        )
    for fmt in ['jpg', 'svg', 'pdf']:
        fig.savefig(f'{model_explain_path}/SME_explain-skip-latent-{key}.{fmt}', dpi=360, bbox_inches='tight')

# %%



