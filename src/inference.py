import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, Optional, Tuple, List
from tqdm import tqdm
import yaml

from src.models import MS_APPT, ESM2EmbeddingExtractor
from src.models.interpretability import AttentionExtractor, BiologicalInterpreter, visualize_attention_matrix, visualize_residue_importance
from src.data import ProteinDataPreprocessor, ProteinPairDataset, collate_protein_pairs
from src.evaluation import calculate_all_metrics, print_metrics_summary, create_performance_report

logger = logging.getLogger(__name__)


class MS_APPT_Predictor:
    def __init__(self, config_path: str, checkpoint_path: str, device: str = 'cuda'):
        self.device = device
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.preprocessor = ProteinDataPreprocessor(self.config)
        
        self._load_normalization_params(checkpoint_path)
        
        logger.info("Initializing ESM-2 embedding extractor...")
        self.embedding_extractor = ESM2EmbeddingExtractor(
            model_name=self.config['model']['encoder']['model_name'],
            cache_dir=self.config['data']['cache_dir'],
            device=device
        )
        
        logger.info("Loading MS-APPT model...")
        self.model = MS_APPT(self.config).to(device)
        self._load_checkpoint(checkpoint_path)
        self.model.eval()
        
    def _load_normalization_params(self, checkpoint_path: str):
        checkpoint_dir = Path(checkpoint_path).parent
        norm_params_path = checkpoint_dir / 'normalization_params.json'
        
        if norm_params_path.exists():
            self.preprocessor.load_normalization_params(str(norm_params_path))
        else:
            logger.warning("Normalization parameters not found. Will need to fit on data.")
    
    def _load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model from checkpoint: {checkpoint_path}")
        
        if 'metrics' in checkpoint:
            logger.info("Checkpoint metrics:")
            for metric, value in checkpoint['metrics'].items():
                if isinstance(value, float):
                    logger.info(f"  {metric}: {value:.4f}")
    
    def _get_embeddings_batch(self, protein1_sequences: List[str],
                            protein2_sequences: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        
        embeddings1 = self.embedding_extractor.get_batch_embeddings(protein1_sequences)
        embeddings2 = self.embedding_extractor.get_batch_embeddings(protein2_sequences)
        
        max_len1 = max(e.shape[0] for e in embeddings1)
        max_len2 = max(e.shape[0] for e in embeddings2)
        
        batch_size = len(embeddings1)
        padded_embeddings1 = torch.zeros(batch_size, max_len1, embeddings1[0].shape[1])
        padded_embeddings2 = torch.zeros(batch_size, max_len2, embeddings2[0].shape[1])
        
        for i, (emb1, emb2) in enumerate(zip(embeddings1, embeddings2)):
            padded_embeddings1[i, :emb1.shape[0]] = emb1
            padded_embeddings2[i, :emb2.shape[0]] = emb2
        
        return padded_embeddings1.to(self.device), padded_embeddings2.to(self.device)
    
    def predict_dataframe(self, df: pd.DataFrame, 
                         has_labels: bool = False,
                         batch_size: Optional[int] = None) -> pd.DataFrame:
        
        if batch_size is None:
            batch_size = self.config['inference']['batch_size']
        
        if self.preprocessor.normalization_params is None and has_labels:
            logger.info("Fitting normalization parameters on provided data...")
            self.preprocessor.normalize_pkd(df['pkd'].values, fit=True)
        
        dataset = ProteinPairDataset(df)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_protein_pairs,
            num_workers=self.config['hardware']['num_workers'],
            pin_memory=self.config['hardware']['pin_memory']
        )
        
        all_predictions = []
        all_indices = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                protein1_embeddings, protein2_embeddings = self._get_embeddings_batch(
                    batch['protein1_sequences'], batch['protein2_sequences']
                )
                
                predictions = self.model(
                    protein1_embeddings, protein2_embeddings,
                    batch['protein1_sequences'], batch['protein2_sequences']
                )
                
                all_predictions.extend(predictions.cpu().numpy())
                all_indices.extend(batch['indices'].cpu().numpy())
        
        predictions_normalized = np.array(all_predictions)
        predictions_denormalized = self.preprocessor.denormalize_pkd(predictions_normalized)
        
        sorted_indices = np.argsort(all_indices)
        predictions_denormalized = predictions_denormalized[sorted_indices]
        
        result_df = df.copy()
        result_df['pkd_predicted'] = predictions_denormalized
        
        if has_labels and 'pkd' in df.columns:
            metrics = calculate_all_metrics(
                df['pkd'].values,
                predictions_denormalized,
                denormalize_fn=None
            )
            print_metrics_summary(metrics)
        
        return result_df
    
    def predict_file(self, input_path: str, output_path: str,
                    has_labels: bool = False) -> Dict:
        
        logger.info(f"Loading data from {input_path}")
        df = pd.read_csv(input_path)
        
        logger.info(f"Preprocessing {len(df)} samples...")
        if has_labels:
            df = self.preprocessor.preprocess_dataset(df, fit_normalization=False)
        else:
            df['protein1_sequence'] = df['protein1_sequence'].apply(self.preprocessor.clean_sequence)
            df['protein2_sequence'] = df['protein2_sequence'].apply(self.preprocessor.clean_sequence)
            
            valid_mask = (
                df['protein1_sequence'].apply(self.preprocessor.validate_sequence) &
                df['protein2_sequence'].apply(self.preprocessor.validate_sequence)
            )
            
            if not valid_mask.all():
                logger.warning(f"Removing {(~valid_mask).sum()} invalid sequences")
                df = df[valid_mask].copy()
        
        logger.info("Running predictions...")
        result_df = self.predict_dataframe(df, has_labels=has_labels)
        
        output_df = result_df[['protein1_sequence', 'protein2_sequence', 'pkd_predicted']].copy()
        if has_labels and 'pkd' in result_df.columns:
            output_df['pkd_actual'] = result_df['pkd']
            output_df['error'] = output_df['pkd_predicted'] - output_df['pkd_actual']
        
        output_df.to_csv(output_path, index=False)
        logger.info(f"Saved predictions to {output_path}")
        
        results = {
            'num_samples': len(result_df),
            'output_path': output_path
        }
        
        if has_labels and 'pkd' in result_df.columns:
            seq_lengths1 = result_df['protein1_sequence'].str.len().values
            seq_lengths2 = result_df['protein2_sequence'].str.len().values
            
            performance_report = create_performance_report(
                result_df['pkd'].values,
                result_df['pkd_predicted'].values,
                sequence_lengths=(seq_lengths1, seq_lengths2)
            )
            results['performance'] = performance_report
        
        return results
    
    def predict_single(self, protein1_sequence: str, protein2_sequence: str,
                      return_attention: bool = False) -> Dict:
        
        protein1_sequence = self.preprocessor.clean_sequence(protein1_sequence)
        protein2_sequence = self.preprocessor.clean_sequence(protein2_sequence)
        
        if not (self.preprocessor.validate_sequence(protein1_sequence) and
                self.preprocessor.validate_sequence(protein2_sequence)):
            raise ValueError("Invalid protein sequence(s)")
        
        result = {}
        
        with torch.no_grad():
            embeddings1 = self.embedding_extractor.get_embedding(protein1_sequence).unsqueeze(0)
            embeddings2 = self.embedding_extractor.get_embedding(protein2_sequence).unsqueeze(0)
            
            embeddings1 = embeddings1.to(self.device)
            embeddings2 = embeddings2.to(self.device)
            
            if return_attention:
                # Use attention extractor
                extractor = AttentionExtractor(self.model, self.device)
                attention_weights = extractor.extract_attention_weights(
                    embeddings1, embeddings2,
                    [protein1_sequence], [protein2_sequence]
                )
                extractor.remove_hooks()
                
                # Analyze interface
                if 'cross_attention' in attention_weights:
                    interface_analysis = extractor.analyze_binding_interface(
                        attention_weights['cross_attention'],
                        protein1_sequence, protein2_sequence
                    )
                    result['interface_analysis'] = interface_analysis
                    result['attention_weights'] = attention_weights
            
            prediction = self.model(
                embeddings1, embeddings2,
                [protein1_sequence], [protein2_sequence]
            )
            
            prediction_normalized = prediction.cpu().numpy()[0]
            prediction_denormalized = self.preprocessor.denormalize_pkd(
                np.array([prediction_normalized])
            )[0]
        
        result['pkd'] = float(prediction_denormalized)
        result['kd'] = 10**(-prediction_denormalized)
        
        if return_attention and 'interface_analysis' in result:
            interpreter = BiologicalInterpreter()
            result['biological_report'] = interpreter.generate_interface_report(
                protein1_sequence, protein2_sequence,
                result['interface_analysis'],
                result['pkd']
            )
        
        return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='MS-APPT Inference')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config.yml file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input CSV file')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to output CSV file')
    parser.add_argument('--has-labels', action='store_true',
                       help='Whether input file contains pkd labels')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    predictor = MS_APPT_Predictor(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    results = predictor.predict_file(
        input_path=args.input,
        output_path=args.output,
        has_labels=args.has_labels
    )
    
    if 'performance' in results:
        report_path = Path(args.output).with_suffix('.report.json')
        with open(report_path, 'w') as f:
            json.dump(results['performance'], f, indent=2)
        logger.info(f"Saved performance report to {report_path}")


if __name__ == '__main__':
    main()