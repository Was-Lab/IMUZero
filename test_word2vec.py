from gensim.models import Word2Vec
import numpy as np
from sklearn.preprocessing import normalize
import scipy.io as sio
# 读取类别名称
def read_classes(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f]
    return classes


# 载入特征描述并按类别分组
def load_features(filename, num_groups):
    with open(filename, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f]
    return classes


# 创建Word2Vec模型
def create_word2vec_model(sentences, vector_size=25, window=5, min_count=1, workers=4):
    model = Word2Vec(sentences=sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    return model


# 将特征文本转换为向量
def feature_texts_to_vectors(model, grouped_texts):
    vectors = []
    for texts in grouped_texts:
        group_vectors = [model.wv[text] for text in texts if text in model.wv]  # 在字典中的单词
        if group_vectors:
            # 获得所有有效向量的均值
            avg_vector = np.mean(group_vectors, axis=0)
        else:
            # 如果这类没有任何有效的单词，则返回零向量
            avg_vector = np.zeros(model.vector_size)
        vectors.append(avg_vector)
    return np.array(vectors)

# 主程序
def main():


    # # 创建一个 225 行，25 列的零矩阵
    matrix = np.zeros((100, 25))
    # # 对每一列填充9个连续的1
    for j in range(25):
        start_index = (j * 4) % 100
        matrix[start_index:start_index + 4, j] = 1

    print(matrix[:,0])

    data = sio.loadmat('../data/xlsa17/data/wurenji/att_wurenji_splits.mat')

    matrix=matrix.astype(np.float64)
    data['original_att'] = matrix

    normalized_feature_vectors = normalize(matrix, norm='l2')
    data['att'] = normalized_feature_vectors

    sio.savemat('../data/xlsa17/data/wurenji/att_wurenji_splits.mat', data)

    # 以下函数需要您自己实现或修改///////////////////////////////////////////////////////////////////////
    classes = read_classes('allclasses.txt')
    features = load_features('feature_part.txt', len(classes))


    # Flatten features for training
    all_features = [feat for sublist in features for feat in sublist]

    model = create_word2vec_model(all_features, vector_size=25)

    feature_vectors = feature_texts_to_vectors(model, features)

    print("The shape of the feature vector matrix is:", feature_vectors.shape)

    data = sio.loadmat('att_wurenji_splits.mat')

    # 创建新属性
    original_att = feature_vectors  # 生成一个随机的(225, 25)矩阵，你可以替换成你想要的数据

    # 将新属性添加到MAT文件中
    data['original_att'] = original_att

    normalized_feature_vectors = normalize(feature_vectors, norm='l2')

    print("The shape of the feature vector L2 matrix is:", normalized_feature_vectors.shape)

    data['att'] = normalized_feature_vectors
    # 保存MAT文件
    sio.savemat('att_wurenji_splits.mat', data)

    matcontent = sio.loadmat('att_wurenji_splits.mat')

    print(matcontent)


if __name__ == "__main__":
    main()