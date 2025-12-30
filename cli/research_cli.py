import argparse
import sys
# Assuming standard module structure based on other files
from research_ingestion import ResearchDataIngestor
from research_ingestion.auto_retrainer import ResearchAutoUpdater
from model_registry import ModelRegistry

def research_ingest_cmd(args):
    
    ingestor = ResearchDataIngestor()
    
    if args.arxiv:
        entry = ingestor.ingest_arxiv(args.arxiv)
        print(f"✅ arXiv article added: {entry['id']}")
    
    if args.pdf:
        entry = ingestor.ingest_pdf(args.pdf)
        print(f"✅ PDF added: {entry['id']}")
    
    if args.csv:
        entry = ingestor.ingest_csv(args.csv)
        print(f"✅ CSV added: {entry['id']}")

def research_train_cmd(args):
    
    def training_callback(parent_id, corpus_path, citation):
        print(f"🔧 Training starting: {corpus_path}")
        return f"trained_model_{corpus_path.split('/')[-1].replace('.txt', '')}"
    
    updater = ResearchAutoUpdater()
    updater.run_once(training_callback)
    print("✅ Automatic training completed!")

def setup_research_parser(subparsers):
    
    ingest_parser = subparsers.add_parser('ingest', help='Add new research data')
    ingest_parser.add_argument('--arxiv', help='arXiv ID')
    ingest_parser.add_argument('--pdf', help='PDF file path')
    ingest_parser.add_argument('--csv', help='CSV file path')
    ingest_parser.set_defaults(func=research_ingest_cmd)
    
    train_parser = subparsers.add_parser('train-auto', help='Start automatic training')
    train_parser.set_defaults(func=research_train_cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Research Pipeline CLI")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    setup_research_parser(subparsers)
    
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()