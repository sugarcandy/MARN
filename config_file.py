config = {}
config['seed'] = [2023]  # Pheme为2023，我们的是2023
config['epochs'] = [60]  # 原始为20，我们的是60
# config['epochs'] = [20]
config['batch_size'] = [64]
config['use_stopwords'] = [True]
config['maxlen'] = [50]
config['ratio'] = [[70, 10, 20]]
config['kernel_sizes'] = [[3, 4, 5]]
config['dropout'] = [0.6]
config['user_self_attention'] = [False]  # 为False
config['n_heads'] = [8]
config['nb_heads'] = [4]
config['num_classes'] = [2]
config['target_names'] = [['NR', 'FR']]

## 多模态融合参数
config['fusion_dim'] = [[300, 300, 300]]
config['n_hidden'] = [24]
config['n_head'] = [4]  ## config['n_head'] = [4]
config['n_layer'] = [1]
# config['fusion_dropout'] = [0.3]  ## 改成0.6试试
config['fusion_dropout'] = [0.3]  ## 改成0.6试试
# 模态数3
config['modal_num'] = [3]
# 多目标优化
config['normalization_type'] = ['loss+']
config['tasks'] = [['classify', 'constractive', 'align']]
