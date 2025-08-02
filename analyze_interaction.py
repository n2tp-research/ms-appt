#!/usr/bin/env python3
"""
Analyze protein-protein interaction with biological interpretation.
Provides detailed analysis of binding interfaces and hotspot residues.
"""

import argparse
import logging
import sys
import json
from pathlib import Path
import torch

from src.inference import MS_APPT_Predictor
from src.models.interpretability import visualize_attention_matrix, visualize_residue_importance


def main():
    parser = argparse.ArgumentParser(
        description='Analyze protein-protein interaction with MS-APPT',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze interaction from sequences
  python analyze_interaction.py --checkpoint model.pt --seq1 MKTAYI... --seq2 MGSSHH...
  
  # Analyze from FASTA file
  python analyze_interaction.py --checkpoint model.pt --fasta proteins.fasta
  
  # Save visualizations
  python analyze_interaction.py --checkpoint model.pt --seq1 ... --seq2 ... --save-plots
        """
    )
    
    parser.add_argument('--config', type=str, default='config.yml',
                       help='Path to config.yml file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--sequences', nargs=2, metavar=('SEQ1', 'SEQ2'),
                           help='Two protein sequences to analyze')
    input_group.add_argument('--fasta', type=str,
                           help='FASTA file with two sequences')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='analysis_output',
                       help='Directory for output files')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save visualization plots')
    parser.add_argument('--save-json', action='store_true',
                       help='Save detailed results as JSON')
    
    # Analysis options
    parser.add_argument('--threshold', type=float, default=0.1,
                       help='Threshold for interface detection (default: 0.1)')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Number of top hotspot residues to show (default: 10)')
    
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Load sequences
    if args.sequences:
        protein1_sequence, protein2_sequence = args.sequences
        seq_names = ['Protein1', 'Protein2']
    else:
        # Load from FASTA
        sequences = []
        seq_names = []
        with open(args.fasta, 'r') as f:
            current_seq = []
            for line in f:
                if line.startswith('>'):
                    if current_seq:
                        sequences.append(''.join(current_seq))
                    seq_names.append(line[1:].strip())
                    current_seq = []
                else:
                    current_seq.append(line.strip())
            if current_seq:
                sequences.append(''.join(current_seq))
        
        if len(sequences) != 2:
            logger.error(f"FASTA file must contain exactly 2 sequences, found {len(sequences)}")
            sys.exit(1)
        
        protein1_sequence, protein2_sequence = sequences
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize predictor
    logger.info(f"Loading model from {args.checkpoint}")
    predictor = MS_APPT_Predictor(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    # Run prediction with attention analysis
    logger.info("Running prediction and interface analysis...")
    result = predictor.predict_single(
        protein1_sequence,
        protein2_sequence,
        return_attention=True
    )
    
    # Display results
    print("\n" + "="*70)
    print("MS-APPT INTERACTION ANALYSIS RESULTS")
    print("="*70 + "\n")
    
    print(f"{seq_names[0]}: {len(protein1_sequence)} aa")
    print(f"{seq_names[1]}: {len(protein2_sequence)} aa")
    print()
    
    print(f"Predicted Binding Affinity:")
    print(f"  pKd: {result['pkd']:.2f}")
    print(f"  Kd: {result['kd']:.2e} M")
    print()
    
    # Print biological interpretation
    if 'biological_report' in result:
        print(result['biological_report'])
    
    # Save visualizations if requested
    if args.save_plots and 'interface_analysis' in result:
        logger.info("Creating visualizations...")
        
        # Attention matrix heatmap
        attention_plot_path = output_dir / 'attention_matrix.png'
        visualize_attention_matrix(
            result['interface_analysis']['attention_matrix'],
            protein1_sequence,
            protein2_sequence,
            save_path=str(attention_plot_path)
        )
        
        # Residue importance plots
        importance1_path = output_dir / f'{seq_names[0]}_importance.png'
        visualize_residue_importance(
            protein1_sequence,
            result['interface_analysis']['protein1_importance'],
            title=f'{seq_names[0]} - Residue Importance for Binding',
            save_path=str(importance1_path)
        )
        
        importance2_path = output_dir / f'{seq_names[1]}_importance.png'
        visualize_residue_importance(
            protein2_sequence,
            result['interface_analysis']['protein2_importance'],
            title=f'{seq_names[1]} - Residue Importance for Binding',
            save_path=str(importance2_path)
        )
        
        logger.info(f"Visualizations saved to {output_dir}")
    
    # Save JSON results if requested
    if args.save_json:
        json_path = output_dir / 'analysis_results.json'
        
        # Prepare JSON-serializable data
        json_data = {
            'sequences': {
                seq_names[0]: protein1_sequence,
                seq_names[1]: protein2_sequence
            },
            'predictions': {
                'pkd': result['pkd'],
                'kd': result['kd']
            }
        }
        
        if 'interface_analysis' in result:
            interface = result['interface_analysis']
            json_data['interface_analysis'] = {
                'protein1_interface_positions': interface['interface1_positions'].tolist(),
                'protein2_interface_positions': interface['interface2_positions'].tolist(),
                'protein1_importance_scores': interface['protein1_importance'].tolist(),
                'protein2_importance_scores': interface['protein2_importance'].tolist()
            }
        
        if 'biological_report' in result:
            json_data['biological_report'] = result['biological_report']
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        logger.info(f"Detailed results saved to {json_path}")
    
    print(f"\nAnalysis complete. Results saved to {output_dir}")


if __name__ == '__main__':
    main()