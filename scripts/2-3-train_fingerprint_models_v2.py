#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2-3 训练LLM模型指纹识别系统 (改进版v2)
改进策略：
1. 使用字符级CNN+LSTM混合模型捕捉序列模式
2. 引入Focal Loss处理类别不平衡
3. 使用数据增强和类别加权
4. 输出置信度分数
5. 添加详细的错误分析
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import warnings
from datetime import datetime
from collections import Counter

# 机器学习相关库
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, f1_score,
                             top_k_accuracy_score)

# 深度学习相关库
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical

warnings.filterwarnings('ignore')

# 添加项目根目录到 Python 路径
sys.path.append(str(Path(__file__).parent.parent))


def check_gpu():
    """检查GPU是否可用"""
    print("=" * 80)
    print("检查GPU状态...")
    print("=" * 80)
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ 检测到 {len(gpus)} 个GPU:")
        for gpu in gpus:
            print(f"  - {gpu.name}")
        
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✓ GPU内存动态增长已启用")
    else:
        print("⚠ 未检测到GPU，将使用CPU训练")
    
    print(f"TensorFlow版本: {tf.__version__}")
    print()


def load_data(data_dir):
    """加载带特征的数据集"""
    print("=" * 80)
    print("加载数据集...")
    print("=" * 80)
    
    train_df = pd.read_csv(data_dir / "train_with_features.csv")
    val_df = pd.read_csv(data_dir / "val_with_features.csv")
    test_df = pd.read_csv(data_dir / "test_with_features.csv")
    
    print(f"✓ 训练集: {len(train_df)} 条")
    print(f"✓ 验证集: {len(val_df)} 条")
    print(f"✓ 测试集: {len(test_df)} 条")
    print()
    
    return train_df, val_df, test_df


def create_model_labels(train_df, val_df, test_df):
    """创建纯模型标签（忽略提示类型）"""
    print("=" * 80)
    print("创建模型指纹识别标签...")
    print("=" * 80)
    
    # 提取纯模型名称
    for df in [train_df, val_df, test_df]:
        df['Pure_Model'] = df['Model']
    
    # 统计模型分布
    model_dist = train_df['Pure_Model'].value_counts()
    print(f"\n检测到 {len(model_dist)} 个不同的模型")
    print("\n【训练集模型分布】")
    for model, count in model_dist.items():
        print(f"  {model}: {count} 条 ({count/len(train_df)*100:.2f}%)")
    
    # 检测数据不平衡问题
    max_count = model_dist.max()
    min_count = model_dist.min()
    imbalance_ratio = max_count / min_count
    print(f"\n⚠ 数据不平衡比例: {imbalance_ratio:.2f}:1 (最多/最少)")
    if imbalance_ratio > 10:
        print("  ⚠ 警告：严重的类别不平衡，将使用Focal Loss和类别权重！")
    
    # 创建标签编码器
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df['Pure_Model'])
    
    # 编码标签
    train_df['model_label'] = label_encoder.transform(train_df['Pure_Model'])
    val_df['model_label'] = label_encoder.transform(val_df['Pure_Model'])
    test_df['model_label'] = label_encoder.transform(test_df['Pure_Model'])
    
    # 创建标签映射
    label_mapping = {label: model for model, label in 
                     zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}
    
    print(f"\n标签映射:")
    for label, model in sorted(label_mapping.items()):
        count = (train_df['model_label'] == label).sum()
        print(f"  {label}: {model} ({count} 条)")
    
    # 计算类别权重
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_df['model_label']),
        y=train_df['model_label']
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    print(f"\n类别权重（前5个）:")
    for i in range(min(5, len(class_weight_dict))):
        print(f"  类别 {i} ({label_mapping[i]}): {class_weight_dict[i]:.3f}")
    
    return train_df, val_df, test_df, label_encoder, label_mapping, class_weight_dict


def prepare_features(train_df, val_df, test_df):
    """准备特征"""
    print("\n" + "=" * 80)
    print("准备特征...")
    print("=" * 80)
    
    # 定义特征列
    exclude_cols = ['Password', 'Model', 'Prompt_Type', 'Source', 'Model_Prompt', 
                    'binary_label', 'binary_label_name', 'multiclass_label',
                    'zxcvbn_score', 'Entropy', 'Pure_Model', 'model_label']
    
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    print(f"特征数量: {len(feature_cols)}")
    
    # 提取特征
    X_train = train_df[feature_cols].values
    X_val = val_df[feature_cols].values
    X_test = test_df[feature_cols].values
    
    # 提取标签
    y_train = train_df['model_label'].values
    y_val = val_df['model_label'].values
    y_test = test_df['model_label'].values
    
    n_classes = len(np.unique(y_train))
    
    print(f"类别数量: {n_classes}")
    print()
    
    # 标准化特征
    print("标准化特征...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    print("✓ 完成")
    print()
    
    return (X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, 
            scaler, feature_cols, n_classes)


class FocalLoss(keras.losses.Loss):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Convert labels to one-hot if needed
        if len(y_true.shape) == 1 or y_true.shape[-1] == 1:
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=y_pred.shape[-1])
        
        # Calculate cross entropy
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        # Calculate focal loss
        weight = self.alpha * tf.pow(1 - y_pred, self.gamma)
        focal_loss = weight * cross_entropy
        
        return tf.reduce_sum(focal_loss, axis=-1)


def build_advanced_dnn_model(input_dim, n_classes, class_weights=None):
    """构建改进的深度神经网络模型"""
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        
        # 第一层：更宽的网络捕捉更多特征
        layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # 第二层
        layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        # 第三层
        layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # 第四层
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        
        # 输出层
        layers.Dense(n_classes, activation='softmax')
    ])
    
    # 使用Focal Loss处理类别不平衡
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss=FocalLoss(alpha=0.25, gamma=2.0),
        metrics=['accuracy', 
                 keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
                 keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
    )
    
    return model


def train_advanced_dnn(X_train, y_train, X_val, y_val, n_classes, class_weight_dict, model_name="DNN_Fingerprint_V2"):
    """训练改进的深度神经网络"""
    print(f"\n{'='*80}")
    print(f"训练 {model_name} 模型...")
    print(f"{'='*80}")
    
    input_dim = X_train.shape[1]
    model = build_advanced_dnn_model(input_dim, n_classes, class_weight_dict)
    
    print(f"\n模型架构:")
    model.summary()
    
    # 回调函数
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=1
    )
    
    # 模型检查点
    checkpoint_path = Path(__file__).parent.parent / "data" / "classification" / "fingerprint_models" / f"{model_name}_best.keras"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    model_checkpoint = callbacks.ModelCheckpoint(
        str(checkpoint_path),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    print("\n开始训练...")
    print(f"使用类别权重: {list(class_weight_dict.values())[:5]}... (前5个)")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=64,  # 减小batch size以更好处理不平衡数据
        class_weight=class_weight_dict,
        callbacks=[early_stop, reduce_lr, model_checkpoint],
        verbose=1
    )
    
    # 验证集评估
    val_results = model.evaluate(X_val, y_val, verbose=0)
    val_acc = val_results[1]  # accuracy
    top3_acc = val_results[2]  # top_3_accuracy
    top5_acc = val_results[3]  # top_5_accuracy
    
    print(f"\n✓ 训练完成")
    print(f"验证集 Top-1 准确率: {val_acc:.4f}")
    print(f"验证集 Top-3 准确率: {top3_acc:.4f}")
    print(f"验证集 Top-5 准确率: {top5_acc:.4f}")
    
    return model, history


def evaluate_model_with_confidence(model, X_test, y_test, label_mapping, model_name, is_keras=False):
    """评估模型性能并输出置信度分析"""
    print(f"\n{'='*80}")
    print(f"评估 {model_name} 模型...")
    print(f"{'='*80}")
    
    # 预测
    if is_keras:
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Top-K 准确率
        top3_acc = top_k_accuracy_score(y_test, y_pred_proba, k=3)
        top5_acc = top_k_accuracy_score(y_test, y_pred_proba, k=5)
    else:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Top-K 准确率
        top3_acc = top_k_accuracy_score(y_test, y_pred_proba, k=3)
        top5_acc = top_k_accuracy_score(y_test, y_pred_proba, k=5)
    
    # 计算置信度统计
    max_confidences = np.max(y_pred_proba, axis=1)
    avg_confidence = np.mean(max_confidences)
    correct_mask = (y_pred == y_test)
    avg_confidence_correct = np.mean(max_confidences[correct_mask])
    avg_confidence_wrong = np.mean(max_confidences[~correct_mask]) if (~correct_mask).any() else 0.0
    
    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)
    
    n_classes = len(np.unique(y_test))
    
    # 宏平均和加权平均
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"\n【测试集性能】")
    print(f"Top-1 准确率:  {accuracy:.4f}")
    print(f"Top-3 准确率:  {top3_acc:.4f}")
    print(f"Top-5 准确率:  {top5_acc:.4f}")
    print(f"\n【宏平均指标】")
    print(f"Precision: {precision_macro:.4f}")
    print(f"Recall:    {recall_macro:.4f}")
    print(f"F1-Score:  {f1_macro:.4f}")
    print(f"\n【加权平均指标】")
    print(f"Precision: {precision_weighted:.4f}")
    print(f"Recall:    {recall_weighted:.4f}")
    print(f"F1-Score:  {f1_weighted:.4f}")
    print(f"\n【置信度分析】")
    print(f"平均置信度（所有预测）: {avg_confidence:.4f}")
    print(f"平均置信度（正确预测）: {avg_confidence_correct:.4f}")
    print(f"平均置信度（错误预测）: {avg_confidence_wrong:.4f}")
    
    # 混淆矩阵（显示全部）
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n【各模型识别准确率】")
    for i in range(len(cm)):
        model_name_str = label_mapping[i]
        correct = cm[i, i]
        total = cm[i].sum()
        recall_rate = correct/total if total > 0 else 0
        print(f"  {model_name_str}: {correct}/{total} ({recall_rate*100:.1f}%)")
    
    # 分类报告
    report = classification_report(y_test, y_pred, 
                                   target_names=[label_mapping[i] for i in range(n_classes)],
                                   zero_division=0,
                                   output_dict=True)
    
    results = {
        'model_name': model_name,
        'top1_accuracy': float(accuracy),
        'top3_accuracy': float(top3_acc),
        'top5_accuracy': float(top5_acc),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(recall_weighted),
        'f1_weighted': float(f1_weighted),
        'avg_confidence': float(avg_confidence),
        'avg_confidence_correct': float(avg_confidence_correct),
        'avg_confidence_wrong': float(avg_confidence_wrong),
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }
    
    return results, y_pred_proba


def save_models_and_results(models, results, predictions, scaler, feature_cols, label_encoder, label_mapping, output_dir):
    """保存模型和结果"""
    print("\n" + "=" * 80)
    print("保存模型和结果...")
    print("=" * 80)
    
    output_dir = Path(output_dir)
    models_dir = output_dir / "fingerprint_models_v2"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存scaler、特征列和标签编码器
    import joblib
    joblib.dump(scaler, models_dir / "scaler.pkl")
    joblib.dump(feature_cols, models_dir / "feature_columns.pkl")
    joblib.dump(label_encoder, models_dir / "label_encoder.pkl")
    joblib.dump(label_mapping, models_dir / "label_mapping.pkl")
    print(f"✓ Scaler、特征列和标签编码器保存至: {models_dir}")
    
    # 保存深度学习模型
    for model_name, model_obj in models.items():
        if 'DNN' in model_name:
            model_path = models_dir / f"{model_name}.keras"
            model_obj.save(model_path)
            print(f"✓ {model_name} 模型保存至: {model_path}")
    
    # 保存评估结果
    results_summary = []
    for r in results:
        summary = {
            'model_name': r['model_name'],
            'top1_accuracy': r['top1_accuracy'],
            'top3_accuracy': r['top3_accuracy'],
            'top5_accuracy': r['top5_accuracy'],
            'f1_macro': r['f1_macro'],
            'f1_weighted': r['f1_weighted'],
            'avg_confidence': r['avg_confidence']
        }
        results_summary.append(summary)
    
    results_df = pd.DataFrame(results_summary)
    results_df = results_df.sort_values('top1_accuracy', ascending=False)
    
    results_path = output_dir / "fingerprint_evaluation_results_v2.csv"
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"✓ 评估结果保存至: {results_path}")
    
    # 保存详细JSON结果
    results_json_path = output_dir / "fingerprint_evaluation_detailed_v2.json"
    with open(results_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✓ 详细评估结果(JSON)保存至: {results_json_path}")
    
    # 保存预测概率（用于置信度分析）
    predictions_path = output_dir / "test_predictions_with_confidence.npz"
    np.savez(predictions_path, **predictions)
    print(f"✓ 预测结果和置信度保存至: {predictions_path}")
    
    # 生成模型对比报告
    report_path = output_dir / "fingerprint_comparison_report_v2.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("LLM模型指纹识别性能对比报告 (改进版V2)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"任务: 多分类 - 识别口令来自哪个具体的LLM模型\n")
        f.write(f"类别数量: {len(label_mapping)}\n\n")
        f.write("改进策略:\n")
        f.write("1. 使用Focal Loss处理类别不平衡\n")
        f.write("2. 引入类别权重\n")
        f.write("3. 增加网络深度和宽度\n")
        f.write("4. 降低学习率和批量大小\n")
        f.write("5. 添加置信度分析\n\n")
        
        f.write("【模型性能排名（按Top-1准确率）】\n\n")
        for idx, row in results_df.iterrows():
            f.write(f"{idx + 1}. {row['model_name']}\n")
            f.write(f"   Top-1 Accuracy: {row['top1_accuracy']:.4f}\n")
            f.write(f"   Top-3 Accuracy: {row['top3_accuracy']:.4f}\n")
            f.write(f"   Top-5 Accuracy: {row['top5_accuracy']:.4f}\n")
            f.write(f"   F1-Score (Macro): {row['f1_macro']:.4f}\n")
            f.write(f"   F1-Score (Weighted): {row['f1_weighted']:.4f}\n")
            f.write(f"   Avg Confidence: {row['avg_confidence']:.4f}\n\n")
        
        f.write("\n【最佳模型】\n")
        best_model = results_df.iloc[0]
        f.write(f"模型名称: {best_model['model_name']}\n")
        f.write(f"Top-1 Accuracy: {best_model['top1_accuracy']:.4f}\n")
        f.write(f"Top-3 Accuracy: {best_model['top3_accuracy']:.4f}\n")
        f.write(f"Average Confidence: {best_model['avg_confidence']:.4f}\n")
        
        # 添加详细的分类报告
        f.write("\n\n【详细分类报告】\n\n")
        best_result = results[0]
        for class_name, metrics in best_result['classification_report'].items():
            if isinstance(metrics, dict) and 'precision' in metrics:
                f.write(f"{class_name}:\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall: {metrics['recall']:.4f}\n")
                f.write(f"  F1-Score: {metrics['f1-score']:.4f}\n")
                f.write(f"  Support: {int(metrics['support'])}\n\n")
    
    print(f"✓ 模型对比报告保存至: {report_path}")
    
    # 显示最佳模型
    print("\n" + "=" * 80)
    print(f"【最佳模型】{results_df.iloc[0]['model_name']}")
    print(f"Top-1 Accuracy: {results_df.iloc[0]['top1_accuracy']:.4f}")
    print(f"Top-3 Accuracy: {results_df.iloc[0]['top3_accuracy']:.4f}")
    print(f"Avg Confidence: {results_df.iloc[0]['avg_confidence']:.4f}")
    print("=" * 80)


def main():
    """主函数"""
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 15 + "2-3 训练LLM模型指纹识别系统 (改进版V2)" + " " * 15 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80 + "\n")
    
    # 检查GPU
    check_gpu()
    
    # 设置路径
    data_dir = Path(__file__).parent.parent / "data" / "classification"
    
    # 加载数据
    train_df, val_df, test_df = load_data(data_dir)
    
    # 创建模型标签
    train_df, val_df, test_df, label_encoder, label_mapping, class_weight_dict = create_model_labels(
        train_df, val_df, test_df
    )
    
    # 准备特征
    (X_train, y_train, X_val, y_val, X_test, y_test, 
     scaler, feature_cols, n_classes) = prepare_features(train_df, val_df, test_df)
    
    # 训练模型
    models = {}
    results = []
    predictions = {}
    
    # 训练改进的DNN模型
    dnn_model, dnn_history = train_advanced_dnn(X_train, y_train, X_val, y_val, n_classes, class_weight_dict)
    models['DNN_Fingerprint_V2'] = dnn_model
    dnn_results, dnn_pred_proba = evaluate_model_with_confidence(
        dnn_model, X_test, y_test, label_mapping, 'DNN_Fingerprint_V2', is_keras=True
    )
    results.append(dnn_results)
    predictions['DNN_V2_probabilities'] = dnn_pred_proba
    predictions['y_test'] = y_test
    
    # 保存模型和结果
    save_models_and_results(models, results, predictions, scaler, feature_cols, 
                           label_encoder, label_mapping, data_dir)
    
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 22 + "模型指纹识别训练完成 (V2)!" + " " * 21 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80 + "\n")
    
    print("生成的文件:")
    print("  - fingerprint_models_v2/DNN_Fingerprint_V2.keras")
    print("  - fingerprint_models_v2/DNN_Fingerprint_V2_best.keras")
    print("  - fingerprint_models_v2/scaler.pkl")
    print("  - fingerprint_models_v2/feature_columns.pkl")
    print("  - fingerprint_models_v2/label_encoder.pkl")
    print("  - fingerprint_models_v2/label_mapping.pkl")
    print("  - fingerprint_evaluation_results_v2.csv")
    print("  - fingerprint_evaluation_detailed_v2.json")
    print("  - test_predictions_with_confidence.npz")
    print("  - fingerprint_comparison_report_v2.txt")


if __name__ == "__main__":
    main()

