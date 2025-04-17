# LightNAS 搜索配置：仿 TinyissimoYOLO 风格，适配 MCU，模型目标大小 ~600KB

work_dir = './save_model/tiny_yolo_like_mcu600kb/'
log_level = 'INFO'

""" 图像尺寸设置 """
image_size = 88  # MCU 友好的输入尺寸（推荐 88x88）

model = dict(
    type='CnnNet',
    structure_info=[
        # Stem
        {'class': 'ConvKXBNRELU', 'in': 3, 'out': 16, 'k': 3, 's': 2, 'g': 1},

        # Block 1
        {
            'class': 'SuperResK1DWK1',
            'in': 16, 'out': 32, 'k': 3, 's': 1, 'g': 1,
            'btn': 16, 'with_res': False, 'L': 1,
            'nbitsA': [8, 8, 8], 'nbitsW': [8, 8, 8]
        },

        # Block 2
        {
            'class': 'SuperResK1DWK1',
            'in': 32, 'out': 64, 'k': 3, 's': 2, 'g': 1,
            'btn': 32, 'with_res': False, 'L': 1,
            'nbitsA': [8, 8, 8], 'nbitsW': [8, 8, 8]
        },

        # Block 3
        {
            'class': 'SuperResK1DWK1',
            'in': 64, 'out': 128, 'k': 3, 's': 1, 'g': 1,
            'btn': 64, 'with_res': False, 'L': 1,
            'nbitsA': [8, 8, 8], 'nbitsW': [8, 8, 8]
        },

        # Block 4
        {
            'class': 'SuperResK1DWK1',
            'in': 128, 'out': 256, 'k': 3, 's': 2, 'g': 1,
            'btn': 128, 'with_res': False, 'L': 1,
            'nbitsA': [8, 8, 8], 'nbitsW': [8, 8, 8]
        },
    ]
)


""" 搜索预算配置 """
budgets = [
    dict(type='flops', budget=100_000_000),         # 100M FLOPs，适配 MCU 实时性
    dict(type='model_size', budget=800_000),        # ~800KB 算法层，对应 Flash 空间
    dict(type='max_feature', budget=600_000),       # 600KB SRAM 顶值（可调整）
    dict(type='layers', budget=100),                 # 限制网络深度
]

""" 评分机制配置 """
score = dict(
    type='madnas',
    multi_block_ratio=[1, 2, 4]  # 越向深层，越重要
)

""" 搜索空间配置（DWConv + 简化卷积） """
space = dict(
    type='space_k1dwk1',
    image_size=image_size,
    maximum_channel=256,

    channel_range_list=[
        [16, 16],
        [32, 32],
        [64, 64],
        [128, 128],
    ],

    search_kernel_size_list=[3],
    search_btn_ratio_list=[1.0],
    search_layer_list=[1],
    search_channel_list=[1.0],
    mutate_method_list=['out', 'k', 'btn', 'L'],  # 不加 res / resproj
    nbitsA=[8],
    nbitsW=[8],
)

""" 搜索流程配置 """
search = dict(
    minor_mutation=False,
    minor_iter=100000,
    popu_size=128,
    num_random_nets=20000,
    sync_size_ratio=1.0,
    num_network=1,
)
