# LightNAS 搜索配置：仿 TinyissimoYOLO 风格，适配 MCU，模型目标大小 ~600KB

work_dir = './save_model/tiny_yolo_like_mcu600kb/'
log_level = 'INFO'

""" 图像尺寸设置 """
image_size = 88  # MCU 友好的输入尺寸（推荐 88x88）

""" 模型初始结构（没有残差） """
model = dict(
    type='CnnNet',
    structure_info = [
    # Stage 1 - Conv + Conv (→ MaxPool)
    {'class': 'ConvKXBNRELU', 'in': 3, 'out': 16, 'k': 3, 's': 1, 'nbitsA': 8, 'nbitsW': 8},
    {'class': 'ConvKXBNRELU', 'in': 16, 'out': 16, 'k': 3, 's': 2, 'nbitsA': 8, 'nbitsW': 8},  # MaxPool 等效降采样

    # Stage 2 - Conv + Conv (→ MaxPool)
    {'class': 'ConvKXBNRELU', 'in': 16, 'out': 32, 'k': 3, 's': 1, 'nbitsA': 8, 'nbitsW': 8},
    {'class': 'ConvKXBNRELU', 'in': 32, 'out': 32, 'k': 3, 's': 2, 'nbitsA': 8, 'nbitsW': 8},

    # Stage 3 - Conv + Conv (→ MaxPool)
    {'class': 'ConvKXBNRELU', 'in': 32, 'out': 64, 'k': 3, 's': 1, 'nbitsA': 8, 'nbitsW': 8},
    {'class': 'ConvKXBNRELU', 'in': 64, 'out': 64, 'k': 3, 's': 2, 'nbitsA': 8, 'nbitsW': 8},

    # Stage 4 - Conv + Conv (→ MaxPool)
    {'class': 'ConvKXBNRELU', 'in': 64, 'out': 128, 'k': 3, 's': 1, 'nbitsA': 8, 'nbitsW': 8},
    {'class': 'ConvKXBNRELU', 'in': 128, 'out': 128, 'k': 3, 's': 2, 'nbitsA': 8, 'nbitsW': 8},
]

)



""" 搜索预算配置 """
budgets = [
    dict(type='flops', budget=100_000_000),         # 100M FLOPs，适配 MCU 实时性
    dict(type='model_size', budget=800_000),      # 1M 参数，对应 1MB Flash
    dict(type='max_feature', budget=600_000),       # 约 600KB SRAM 峰值（可以调）
    dict(type='layers', budget=32),                 # 限制网络太深，利于部署
    # dict(type='latency', budget=xxx),             # ❌ 不建议使用
]


""" 评分机制配置 """
score = dict(
    type='madnas',
    multi_block_ratio=[1, 1, 1, 2, 4, 8],  # 更关注靠近输出层的语义
)

""" 搜索空间配置（DWConv + 简化卷积） """
space = dict(
    type='space_k1kx',
    image_size=image_size,
    kernel_size_list=[3],
    channel_range_list=[[16, 64], [16, 64], [16, 64], [32, 128], [32, 128], [64, 192]],
)

""" 搜索流程配置 """
search = dict(
    minor_mutation=False,
    minor_iter=100000,
    popu_size=128,
    num_random_nets=50000,
    sync_size_ratio=1.0,
    num_network=1,
)
