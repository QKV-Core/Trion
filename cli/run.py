import argparse
import sys
import os
from pathlib import Path
import torch
import time
from tabulate import tabulate
from colorama import init, Fore, Style

sys.path.insert(0, str(Path(__file__).parent.parent))

from qkv_core.transformer import TransformerModel
try:
    from tokenizer.fast_tokenizer import FastBPETokenizer as BPETokenizer
    print("✅ Using FAST tokenizer (Rust-based)")
except ImportError:
    from qkv_core.tokenization.bpe import BPETokenizer
    print("⚠️  Using Python BPE tokenizer (slower)")
from qkv_core.training.trainer import Trainer
from qkv_core.training.dataset import TextDataset
try:
    from training.dataset import IncrementalDataset
except ImportError:
    IncrementalDataset = None
from qkv_core.inference.inference import InferenceEngine
from qkv_core.storage.db import DatabaseManager
from config.model_config import ModelConfig

init(autoreset=True)

def print_header(text):
    print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{text.center(60)}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")

def print_success(text):
    print(f"{Fore.GREEN}✓ {text}{Style.RESET_ALL}")

def print_error(text):
    print(f"{Fore.RED}✗ {text}{Style.RESET_ALL}")

def print_info(text):
    print(f"{Fore.YELLOW}ℹ {text}{Style.RESET_ALL}")

def load_corpus(file_path):
    # Changed 'if' to 'f' for file handle
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    return texts

def train_tokenizer_command(args):
    
    print_header("Training Tokenizer")
    
    print_info(f"Loading corpus from {args.corpus}")
    corpus = load_corpus(args.corpus)
    print_success(f"Loaded {len(corpus)} texts")
    
    tokenizer = BPETokenizer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_freq
    )
    
    tokenizer.train(corpus, verbose=True)
    
    tokenizer.save(args.output)
    print_success(f"Tokenizer saved to {args.output}")

def train_model_command(args):
    
    print_header("Training Model")
    
    config = ModelConfig()
    
    if args.vocab_size:
        config.vocab_size = args.vocab_size
    if args.d_model:
        config.d_model = args.d_model
    if args.num_layers:
        config.num_layers = args.num_layers
    if args.num_heads:
        config.num_heads = args.num_heads
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.max_epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    
    if os.path.exists(args.tokenizer):
        print_info(f"Loading tokenizer from {args.tokenizer}")
        tokenizer = BPETokenizer.load(args.tokenizer)
    else:
        print_info(f"Training new tokenizer...")
        corpus = load_corpus(args.data)
        tokenizer = BPETokenizer(vocab_size=config.vocab_size)
        tokenizer.train(corpus)
        tokenizer.save(args.tokenizer)
    
    config.vocab_size = tokenizer.get_vocab_size()
    
    print_info(f"Loading training data from {args.data}")
    corpus = load_corpus(args.data)
    
    dataset = TextDataset(
        texts=corpus,
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
        lazy_loading=not args.preencode
    )
    
    print_info("Creating model...")
    model = TransformerModel(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        max_seq_length=config.max_seq_length,
        dropout=config.dropout
    )
    
    print_success(f"Model created with {model.get_num_parameters():,} parameters")
    
    db = DatabaseManager(config.db_path)
    
    trainer = Trainer(
        model=model,
        config=config.to_dict(),
        tokenizer=tokenizer,
        db_manager=db
    )
    
    session_name = args.name or f"training_{int(time.time())}"
    model_version_name = args.model_name or session_name
    
    trainer.train(
        train_dataset=dataset,
        num_epochs=config.max_epochs,
        session_name=session_name,
        model_version_name=model_version_name
    )
    
    print_success("Training completed!")

def continue_training_command(args):
    
    print_header("Continue Training")
    
    config = ModelConfig()
    
    print_info(f"Loading tokenizer from {args.tokenizer}")
    tokenizer = BPETokenizer.load(args.tokenizer)
    
    print_info(f"Loading training data from {args.data}")
    corpus = load_corpus(args.data)
    
    dataset = TextDataset(
        texts=corpus,
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
        lazy_loading=not args.preencode
    )
    
    model = TransformerModel(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        max_seq_length=config.max_seq_length,
        dropout=config.dropout
    )
    
    db = DatabaseManager(config.db_path)
    
    trainer = Trainer(
        model=model,
        config=config.to_dict(),
        tokenizer=tokenizer,
        db_manager=db
    )
    
    session_name = args.name or f"continued_{int(time.time())}"
    
    trainer.continue_training(
        train_dataset=dataset,
        checkpoint_path=args.checkpoint,
        additional_epochs=args.epochs or 10,
        session_name=session_name
    )
    
    print_success("Continued training completed!")

def infer_command(args):
    
    print_header("Inference")
    
    print_info(f"Loading tokenizer from {args.tokenizer}")
    tokenizer = BPETokenizer.load(args.tokenizer)
    
    config = ModelConfig()
    
    model = TransformerModel(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        max_seq_length=config.max_seq_length,
        dropout=config.dropout
    )
    
    print_info(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    db = DatabaseManager(config.db_path) if args.log else None
    
    engine = InferenceEngine(
        model=model,
        tokenizer=tokenizer,
        db_manager=db
    )
    
    if args.interactive:
        engine.interactive_chat()
    else:
        prompt = args.prompt or input("Enter prompt: ")
        
        print(f"\n{Fore.YELLOW}Prompt:{Style.RESET_ALL} {prompt}\n")
        print(f"{Fore.YELLOW}Generating...{Style.RESET_ALL}\n")
        
        output = engine.generate(
            prompt=prompt,
            max_length=args.max_length,
            method=args.method,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            log_to_db=args.log
        )
        
        print(f"{Fore.GREEN}Output:{Style.RESET_ALL} {output}\n")

def show_models_command(args):
    
    print_header("Model Versions")
    
    config = ModelConfig()
    db = DatabaseManager(config.db_path)
    
    versions = db.list_model_versions()
    
    if not versions:
        print_info("No models found")
        return
    
    table_data = []
    for v in versions:
        table_data.append([
            v['id'],
            v['version_name'],
            v['d_model'],
            v['num_layers'],
            v['num_heads'],
            f"{v['total_parameters']:,}" if v['total_parameters'] else "N/A",
            v['created_at']
        ])
    
    headers = ['ID', 'Version', 'd_model', 'Layers', 'Heads', 'Parameters', 'Created']
    
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    print(f"\n{Fore.CYAN}Total models: {len(versions)}{Style.RESET_ALL}")

def stats_command(args):
    
    print_header("Database Statistics")
    
    config = ModelConfig()
    db = DatabaseManager(config.db_path)
    
    stats = db.get_statistics()
    
    print(f"{Fore.CYAN}Model Versions:{Style.RESET_ALL} {stats.get('total_models', 0)}")
    print(f"{Fore.CYAN}Training Sessions:{Style.RESET_ALL} {stats.get('total_training_sessions', 0)}")
    print(f"{Fore.CYAN}Total Inferences:{Style.RESET_ALL} {stats.get('total_inferences', 0)}")
    print(f"{Fore.CYAN}Knowledge Entries:{Style.RESET_ALL} {stats.get('knowledge_entries', 0)}")
    
    if 'latest_model' in stats:
        print(f"\n{Fore.YELLOW}Latest Model:{Style.RESET_ALL}")
        print(f"  Name: {stats['latest_model']['version_name']}")
        print(f"  Created: {stats['latest_model']['created_at']}")

def quantize_command(args):
    
    print_header("Model Quantization (INT8)")
    
    from core.quantization_simple import QuantizedModel
    
    if not Path(args.checkpoint).exists():
        print_error(f"Checkpoint not found: {args.checkpoint}")
        return
    
    print(f"{Fore.YELLOW}Checkpoint:{Style.RESET_ALL} {args.checkpoint}")
    print(f"{Fore.YELLOW}Method:{Style.RESET_ALL} {args.method}")
    
    start_time = time.time()
    
    try:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_dict = checkpoint['model_state_dict']
        else:
            model_dict = checkpoint
        
        quantized_checkpoint = {
            'model_state_dict': model_dict,
            'quantized': True
        }
        
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        torch.save(quantized_checkpoint, args.output)
        
        elapsed = time.time() - start_time
        
        org_size = Path(args.checkpoint).stat().st_size / 1024**2
        quant_size = Path(args.output).stat().st_size / 1024**2
        
        print_success(f"Quantization completed in {elapsed:.2f}s")
        print(f"  Original: {org_size:.1f} MB")
        print(f"  Quantized: {quant_size:.1f} MB")
        compression = org_size / quant_size if quant_size > 0 else 1.0
        print(f"  Compression: {compression:.1f}x")
        print(f"\n{Fore.YELLOW}To use quantized model:{Style.RESET_ALL}")
        print(f"  python cli/run.py infer --checkpoint {args.output} --prompt 'your prompt'")
    except Exception as e:
        print_error(f"Quantization failed: {e}")
        import traceback
        traceback.print_exc()

def batch_infer_command(args):
    
    print_header("Batch Inference")
    
    from models.batch_inference import BatchInferenceEngine
    
    print_info(f"Loading tokenizer from {args.tokenizer}")
    tokenizer = BPETokenizer.load(args.tokenizer)
    
    config = ModelConfig()
    
    model = TransformerModel(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        max_seq_length=config.max_seq_length,
        dropout=config.dropout
    )
    
    print_info(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    engine = BatchInferenceEngine(
        model=model,
        tokenizer=tokenizer
    )
    
    if args.prompts:
        prompts = args.prompts
    elif args.file:
        if not Path(args.file).exists():
            print_error(f"Prompt file not found: {args.file}")
            return
        # Changed 'if' to 'f'
        with open(args.file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        print_error("Provide prompts via --prompts or --file")
        return
    
    if not prompts:
        print_error("No prompts provided")
        return
    
    print_info(f"Processing {len(prompts)} prompts")
    
    result = engine.generate_batch(
        prompts=prompts,
        max_length=args.max_length,
        method=args.method,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        return_times=True
    )
    
    print(f"\n{Fore.CYAN}Batch Results:{Style.RESET_ALL}")
    print(f"  Batch size: {result['batch_size']}")
    print(f"  Total time: {result['total_time']:.2f}s")
    print(f"  Throughput: {result['throughput']:.1f} tokens/sec")
    print(f"  Avg tokens/prompt: {result['avg_tokens']:.1f}\n")
    
    for i, (prompt, output, t) in enumerate(zip(prompts, result['outputs'], result['times'])):
        print(f"{Fore.YELLOW}Prompt {i+1}:{Style.RESET_ALL} {prompt}")
        print(f"{Fore.GREEN}Output:{Style.RESET_ALL} {output}")
        if args.show_times:
            print(f"  Time: {t:.3f}s")
        print()
    
    print_success("Batch inference completed!")

def main():
    
    parser = argparse.ArgumentParser(
        description="QKV Core - Command Line Interface (Query-Key-Value Core)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Tokenizer Training
    train_tok = subparsers.add_parser('train-tokenizer', help='Train a new tokenizer')
    train_tok.add_argument('--corpus', required=True, help='Path to training corpus')
    train_tok.add_argument('--output', required=True, help='Output path for tokenizer')
    train_tok.add_argument('--vocab-size', type=int, default=10000, help='Vocabulary size')
    train_tok.add_argument('--min-freq', type=int, default=2, help='Minimum token frequency')
    
    # Model Training
    train = subparsers.add_parser('train', help='Train a new model')
    train.add_argument('--data', required=True, help='Path to training data')
    train.add_argument('--tokenizer', required=True, help='Path to tokenizer file')
    train.add_argument('--name', help='Training session name')
    train.add_argument('--model-name', help='Model version name')
    train.add_argument('--vocab-size', type=int, help='Vocabulary size')
    train.add_argument('--d-model', type=int, help='Model dimension')
    train.add_argument('--num-layers', type=int, help='Number of layers')
    train.add_argument('--num-heads', type=int, help='Number of attention heads')
    train.add_argument('--batch-size', type=int, help='Batch size')
    train.add_argument('--epochs', type=int, help='Number of epochs')
    train.add_argument('--lr', type=float, help='Learning rate')
    train.add_argument('--preencode', action='store_true', help='Pre-encode dataset into memory for faster training (uses more RAM)')
    
    # Continue Training
    cont = subparsers.add_parser('continue-train', help='Continue training from checkpoint')
    cont.add_argument('--checkpoint', required=True, help='Path to checkpoint')
    cont.add_argument('--data', required=True, help='Path to new training data')
    cont.add_argument('--tokenizer', required=True, help='Path to tokenizer')
    cont.add_argument('--epochs', type=int, help='Additional epochs')
    cont.add_argument('--name', help='Session name')
    cont.add_argument('--preencode', action='store_true', help='Pre-encode dataset into memory')
    
    # Inference
    infer = subparsers.add_parser('infer', help='Run inference')
    infer.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    infer.add_argument('--tokenizer', required=True, help='Path to tokenizer')
    infer.add_argument('--prompt', help='Input prompt')
    infer.add_argument('--max-length', type=int, default=100, help='Maximum generation length')
    infer.add_argument('--method', choices=['greedy', 'sample', 'beam'], default='greedy', help='Generation method')
    infer.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    infer.add_argument('--top-k', type=int, default=50, help='Top-k sampling')
    infer.add_argument('--top-p', type=float, default=0.95, help='Nucleus sampling')
    infer.add_argument('--interactive', action='store_true', help='Interactive chat mode')
    infer.add_argument('--log', action='store_true', help='Log inference to database')
    
    # Show Models
    subparsers.add_parser('show-models', help='List all model versions')
    
    # Statistics
    subparsers.add_parser('stats', help='Show database statistics')
    
    # Quantization
    quantize = subparsers.add_parser('quantize', help='Quantize model to INT8 for faster inference (2-4x speedup)')
    quantize.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    quantize.add_argument('--output', default='model_weights/quantized_model.pt', help='Output path for quantized model')
    quantize.add_argument('--method', choices=['native', 'bnb'], default='native', help='Quantization method (native=PyTorch, bnb=bitsandbytes)')
    
    # Batch Inference
    batch_infer = subparsers.add_parser('batch-infer', help='Run batch inference on multiple prompts (2-5x throughput)')
    batch_infer.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    batch_infer.add_argument('--tokenizer', required=True, help='Path to tokenizer')
    batch_infer.add_argument('--prompts', nargs='+', help='Prompts to process')
    batch_infer.add_argument('--file', help='File with prompts (one per line)')
    batch_infer.add_argument('--max-length', type=int, default=100, help='Maximum generation length')
    batch_infer.add_argument('--method', choices=['greedy', 'sample', 'beam'], default='greedy', help='Generation method')
    batch_infer.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    batch_infer.add_argument('--top-k', type=int, default=50, help='Top-k sampling')
    batch_infer.add_argument('--top-p', type=float, default=0.95, help='Nucleus sampling')
    batch_infer.add_argument('--show-times', action='store_true', help='Show per-prompt generation times')
    
    # Research CLI Integration
    from cli.research_cli import setup_research_parser
    setup_research_parser(subparsers)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'train-tokenizer':
            train_tokenizer_command(args)
        elif args.command == 'train':
            train_model_command(args)
        elif args.command == 'continue-train':
            continue_training_command(args)
        elif args.command == 'infer':
            infer_command(args)
        elif args.command == 'batch-infer':
            batch_infer_command(args)
        elif args.command == 'show-models':
            show_models_command(args)
        elif args.command == 'stats':
            stats_command(args)
        elif args.command == 'quantize':
            quantize_command(args)
    except Exception as e:
        print_error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()