"""
RoBERTa MLM 预训练执行脚本
用于配置参数并启动预训练
"""

import os
import sys
import argparse
from RoBERTa_Custom.pretrain import main  # 匯入 pretrain.py 的 main 函數

def configure_and_run():
    """
    配置参数并调用预训练主函数
    """
    parser = argparse.ArgumentParser(description='RoBERTa MLM 预训练')
    
    # 基本设置
    parser.add_argument('--data_dir', default="C:/Users/user/Desktop/RoBERTa_Model_Selfdesign/DataSet/cleaned_news_collection.csv",
                        help='清理后的数据集路径')
    parser.add_argument('--output_dir', default="C:/Users/user/Desktop/RoBERTa_Model_Selfdesign/output",
                        help='模型保存目录')
    parser.add_argument('--tokenizer_path', default="hfl/chinese-bert-wwm-ext", 
                        help='使用的 tokenizer 路径')
    
    # 模型配置
    parser.add_argument('--model_size', choices=['small', 'base', 'large'], default='base',
                        help='选择模型大小配置')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16,
                        help='训练批次大小')
    parser.add_argument('--epochs', type=int, default=5,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--seq_len', type=int, default=512,
                        help='最大序列长度')
    parser.add_argument('--save_steps', type=int, default=5000,
                        help='每多少步保存一次模型')
    parser.add_argument('--resume', default=None,
                        help='从检查点恢复训练')
    parser.add_argument('--fp16', action='store_true',
                        help='使用混合精度训练')
    
    args = parser.parse_args()
    
    # 配置预训练参数
    pretrain_args = []
    
    # 基本路径
    pretrain_args.extend(['--data_dir', args.data_dir])
    pretrain_args.extend(['--output_dir', args.output_dir])
    pretrain_args.extend(['--tokenizer_path', args.tokenizer_path])
    
    # 根据模型大小设置模型参数
    if args.model_size == 'small':
        pretrain_args.extend([
            '--hidden_size', '512',
            '--num_heads', '8',
            '--intermediate_size', '2048',
            '--num_hidden_layers', '8'
        ])
    elif args.model_size == 'base':
        pretrain_args.extend([
            '--hidden_size', '768',
            '--num_heads', '12',
            '--intermediate_size', '3072',
            '--num_hidden_layers', '12'
        ])
    elif args.model_size == 'large':
        pretrain_args.extend([
            '--hidden_size', '1024',
            '--num_heads', '16',
            '--intermediate_size', '4096',
            '--num_hidden_layers', '24'
        ])
    
    # 训练参数
    pretrain_args.extend(['--batch_size', str(args.batch_size)])
    pretrain_args.extend(['--num_train_epochs', str(args.epochs)])
    pretrain_args.extend(['--learning_rate', str(args.lr)])
    pretrain_args.extend(['--max_seq_length', str(args.seq_len)])
    pretrain_args.extend(['--save_steps', str(args.save_steps)])
    
    # 恢复训练
    if args.resume:
        pretrain_args.extend(['--resume_from_checkpoint', args.resume])
    
    # 混合精度训练
    if args.fp16:
        pretrain_args.append('--fp16')
    
    # 模擬命令行參數
    sys.argv = [sys.argv[0]] + pretrain_args

    # 调用预训练模块
    main()

if __name__ == "__main__":
    configure_and_run()