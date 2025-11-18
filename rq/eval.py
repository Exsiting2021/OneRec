#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RQ-VAE 模型评估 Pipeline
========================

该脚本提供了一个完整的评估 pipeline，用于评估训练好的 RQ-VAE 模型的各项指标，
包括码本利用率、token 分布熵等关键指标。
"""

import torch
import argparse
import numpy as np
from collections import Counter
from models.rqvae import RQVAE
from data.dataset import EmbeddingDataset
from torch.utils.data import DataLoader
import json
import os


def load_model(model_path, device):
    """加载训练好的模型"""
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    # 创建模型
    model = RQVAE(
        input_dim=config['input_dim'],
        hidden_dims=config['hidden_dims'],
        latent_dim=config['latent_dim'],
        num_codebooks=config['num_codebooks'],
        codebook_size=config['codebook_size'],
        beta=config['beta']
    )
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("Model loaded successfully.")
    return model, config, checkpoint


def compute_reconstruction_error(model, data_loader, device):
    """
    计算重构误差
    
    Args:
        model: RQVAE 模型
        data_loader: 数据加载器
        device: 计算设备
        
    Returns:
        float: 平均重构误差
    """
    print("Computing reconstruction error...")
    total_loss = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            batch = batch.to(device)
            reconstructed, loss, _ = model(batch)
            total_loss += loss.item() * batch.size(0)
            num_samples += batch.size(0)
            
            # 只计算前10个批次以节省时间
            if batch_idx >= 10:
                break
    
    avg_loss = total_loss / num_samples
    print(f"Reconstruction error (MSE): {avg_loss:.6f}")
    return avg_loss


def compute_codebook_utilization(indices, codebook_size):
    """
    计算码本利用率
    
    Args:
        indices: 索引张量，形状为 (batch_size, num_codebooks)
        codebook_size: 码本大小
        
    Returns:
        dict: 利用率统计信息
    """
    batch_size, num_codebooks = indices.shape
    utilization = {}
    
    for codebook_idx in range(num_codebooks):
        # 获取当前码本的所有索引
        codebook_indices = indices[:, codebook_idx].cpu().numpy()
        
        # 统计每个索引的出现次数
        counter = Counter(codebook_indices)
        
        # 计算使用过的码字数量
        used_codes = len(counter)
        total_codes = codebook_size
        
        # 计算利用率
        utilization_rate = used_codes / total_codes
        
        # 计算熵 (衡量分布均匀性)
        counts = np.array(list(counter.values()))
        probs = counts / np.sum(counts)
        entropy = -np.sum(probs * np.log(probs + 1e-8))  # 添加小值避免log(0)
        max_entropy = np.log(total_codes)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        utilization[int(codebook_idx)] = {
            'used_codes': int(used_codes),
            'total_codes': int(total_codes),
            'utilization_rate': float(utilization_rate),
            'entropy': float(entropy),
            'normalized_entropy': float(normalized_entropy),
            'counts': {int(k): int(v) for k, v in counter.items()}  # 确保键和值都是Python原生类型
        }
    
    return utilization


def compute_token_distribution_entropy(indices, codebook_size):
    """
    计算token分布熵
    
    Args:
        indices: 索引张量，形状为 (batch_size, num_codebooks)
        codebook_size: 码本大小
        
    Returns:
        dict: token分布熵统计信息
    """
    batch_size, num_codebooks = indices.shape
    
    # 将所有码本的索引展平为一维数组
    all_indices = indices.view(-1).cpu().numpy()
    total_tokens = len(all_indices)
    
    # 统计每个token的出现次数
    counter = Counter(all_indices)
    
    # 计算每个token的概率
    probs = np.array(list(counter.values())) / total_tokens
    
    # 计算熵
    entropy = -np.sum(probs * np.log(probs + 1e-8))  # 添加小值避免log(0)
    max_entropy = np.log(codebook_size)  # 最大熵基于码本大小
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    # 计算perplexity (困惑度)
    perplexity = np.exp(entropy)
    
    return {
        'entropy': float(entropy),
        'normalized_entropy': float(normalized_entropy),
        'perplexity': float(perplexity),
        'total_tokens': int(total_tokens),
        'unique_tokens': int(len(counter)),
        'codebook_size': int(codebook_size),
        'utilization_rate': float(len(counter) / codebook_size),
        'counts': {int(k): int(v) for k, v in counter.items()}  # 确保键和值都是Python原生类型
    }


def evaluate_model(model, data_loader, device, max_batches=50):
    """
    对模型进行全面评估
    
    Args:
        model: RQVAE 模型
        data_loader: 数据加载器
        device: 计算设备
        max_batches: 最大评估批次数
        
    Returns:
        dict: 评估结果
    """
    print("Starting comprehensive model evaluation...")
    print("=" * 50)
    
    all_indices = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            batch = batch.to(device)
            _, _, indices = model(batch)
            all_indices.append(indices)
            
            if batch_idx >= max_batches:
                break
    
    # 合并所有索引
    all_indices = torch.cat(all_indices, dim=0)
    
    # 计算码本利用率
    print("\n1. Computing codebook utilization...")
    utilization_stats = compute_codebook_utilization(
        all_indices, model.codebooks[0].embedding.num_embeddings
    )
    
    # 计算token分布熵
    print("\n2. Computing token distribution entropy...")
    token_entropy_stats = compute_token_distribution_entropy(
        all_indices, model.codebooks[0].embedding.num_embeddings
    )
    
    # 计算平均统计信息
    total_utilization = sum(stats['utilization_rate'] for stats in utilization_stats.values())
    total_normalized_entropy = sum(stats['normalized_entropy'] for stats in utilization_stats.values())
    
    avg_utilization = total_utilization / len(utilization_stats)
    avg_normalized_entropy = total_normalized_entropy / len(utilization_stats)
    
    # 组织评估结果
    evaluation_result = {
        'codebook_utilization': utilization_stats,
        'token_entropy': token_entropy_stats,
        'summary': {
            'avg_codebook_utilization': float(avg_utilization),
            'avg_normalized_entropy': float(avg_normalized_entropy),
            'num_codebooks': len(utilization_stats)
        }
    }
    
    return evaluation_result


def print_evaluation_report(evaluation_result):
    """
    打印评估报告
    
    Args:
        evaluation_result: 评估结果字典
    """
    print("\n" + "=" * 60)
    print("RQ-VAE MODEL EVALUATION REPORT")
    print("=" * 60)
    
    # 码本利用率详情
    print("\n1. Codebook Utilization Details:")
    print("-" * 40)
    for codebook_idx, stats in evaluation_result['codebook_utilization'].items():
        print(f"Codebook {codebook_idx + 1}:")
        print(f"  Used codes: {stats['used_codes']}/{stats['total_codes']}")
        print(f"  Utilization rate: {stats['utilization_rate']:.2%}")
        print(f"  Entropy: {stats['entropy']:.4f}")
        print(f"  Normalized entropy: {stats['normalized_entropy']:.4f}")
        print()
    
    # Token分布熵
    print("2. Token Distribution Entropy:")
    print("-" * 40)
    token_stats = evaluation_result['token_entropy']
    print(f"Total tokens: {token_stats['total_tokens']}")
    print(f"Unique tokens: {token_stats['unique_tokens']}")
    print(f"Codebook size: {token_stats['codebook_size']}")
    print(f"Token utilization rate: {token_stats['utilization_rate']:.2%}")
    print(f"Token distribution entropy: {token_stats['entropy']:.4f}")
    print(f"Normalized token entropy: {token_stats['normalized_entropy']:.4f}")
    print(f"Perplexity: {token_stats['perplexity']:.4f}")
    
    # 摘要
    print("\n3. Summary:")
    print("-" * 40)
    summary = evaluation_result['summary']
    print(f"Average codebook utilization: {summary['avg_codebook_utilization']:.2%}")
    print(f"Average normalized entropy: {summary['avg_normalized_entropy']:.4f}")
    print(f"Number of codebooks: {summary['num_codebooks']}")


def save_evaluation_result(evaluation_result, output_path):
    """
    保存评估结果到文件
    
    Args:
        evaluation_result: 评估结果字典
        output_path: 输出文件路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_result, f, indent=2, ensure_ascii=False)
    
    print(f"\nEvaluation results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='RQ-VAE Model Evaluation Pipeline')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model file')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the evaluation data file')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for evaluation (default: 256)')
    parser.add_argument('--max_batches', type=int, default=50,
                        help='Maximum number of batches to evaluate (default: 50)')
    parser.add_argument('--output_path', type=str, default='./evaluation_result.json',
                        help='Path to save evaluation results (default: ./evaluation_result.json)')
    
    args = parser.parse_args()
    
    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载模型
    model, config, checkpoint = load_model(args.model_path, device)
    
    # 打印模型配置
    print("\nModel Configuration:")
    print("-" * 30)
    print(f"  Input dimension: {config['input_dim']}")
    print(f"  Latent dimension: {config['latent_dim']}")
    print(f"  Number of codebooks: {config['num_codebooks']}")
    print(f"  Codebook size: {config['codebook_size']}")
    print()
    
    # 加载数据
    print("Loading evaluation data...")
    dataset = EmbeddingDataset(args.data_path)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=2
    )
    print(f"Loaded {len(dataset)} samples for evaluation")
    
    # 执行全面评估
    evaluation_result = evaluate_model(model, dataloader, device, args.max_batches)
    
    # 打印评估报告
    print_evaluation_report(evaluation_result)
    
    # 保存评估结果
    save_evaluation_result(evaluation_result, args.output_path)
    
    print("\nEvaluation pipeline completed successfully!")


if __name__ == "__main__":
    main()