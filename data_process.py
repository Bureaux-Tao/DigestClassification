def load_data(filename):
    """加载数据
    单条格式：(文本, 标签id)
    """
    D = []
    with open(filename, encoding = 'utf-8') as f:
        for l in f:
            text, label = l.strip().strip('\n').split('\t')
            D.append((text, int(label)))
    return D