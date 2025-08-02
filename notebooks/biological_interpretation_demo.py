"""
Demonstration of MS-APPT Biological Interpretation Features

This script shows how to:
1. Make predictions with attention analysis
2. Identify potential binding interfaces
3. Find hotspot residues
4. Generate biological interpretation reports
"""

import sys
sys.path.append('..')

from src.inference import MS_APPT_Predictor
from src.models.interpretability import visualize_attention_matrix, visualize_residue_importance
import matplotlib.pyplot as plt

# Example: Antibody-antigen interaction
# These are example sequences - replace with your actual sequences
antibody_heavy = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAK"
antigen = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR"

def main():
    # Initialize predictor (update paths as needed)
    print("Loading MS-APPT model...")
    predictor = MS_APPT_Predictor(
        config_path='../config.yml',
        checkpoint_path='../checkpoints/best_model.pt',
        device='cuda'
    )
    
    # Run prediction with attention analysis
    print("\nRunning prediction with interface analysis...")
    result = predictor.predict_single(
        antibody_heavy,
        antigen,
        return_attention=True
    )
    
    # Display results
    print("\n" + "="*60)
    print("BIOLOGICAL INTERPRETATION RESULTS")
    print("="*60)
    
    print(f"\nPredicted Binding Affinity:")
    print(f"  pKd: {result['pkd']:.2f}")
    print(f"  Kd: {result['kd']:.2e} M")
    
    # Print biological report
    if 'biological_report' in result:
        print("\n" + result['biological_report'])
    
    # Visualize attention matrix
    if 'interface_analysis' in result:
        print("\nGenerating visualizations...")
        
        # Attention heatmap
        plt.figure(figsize=(12, 10))
        visualize_attention_matrix(
            result['interface_analysis']['attention_matrix'],
            antibody_heavy[:50],  # Show first 50 residues for clarity
            antigen[:50],
            save_path=None
        )
        plt.title("Cross-Attention Matrix\n(Antibody-Antigen Interface)")
        plt.tight_layout()
        plt.show()
        
        # Residue importance for antibody
        plt.figure(figsize=(15, 6))
        visualize_residue_importance(
            antibody_heavy,
            result['interface_analysis']['protein1_importance'],
            title="Antibody - Residue Importance for Binding",
            save_path=None
        )
        plt.tight_layout()
        plt.show()
        
        # Residue importance for antigen
        plt.figure(figsize=(15, 6))
        visualize_residue_importance(
            antigen,
            result['interface_analysis']['protein2_importance'],
            title="Antigen - Residue Importance for Binding",
            save_path=None
        )
        plt.tight_layout()
        plt.show()
        
        # Print top interface residues
        print("\nTop 10 Antibody Interface Residues:")
        interface1_pos = result['interface_analysis']['interface1_positions']
        importance1 = result['interface_analysis']['protein1_importance']
        
        residues1 = [(i+1, antibody_heavy[i], importance1[i]) 
                    for i in interface1_pos if i < len(antibody_heavy)]
        residues1.sort(key=lambda x: x[2], reverse=True)
        
        for pos, res, score in residues1[:10]:
            print(f"  {res}{pos}: {score:.3f}")
        
        print("\nTop 10 Antigen Interface Residues:")
        interface2_pos = result['interface_analysis']['interface2_positions']
        importance2 = result['interface_analysis']['protein2_importance']
        
        residues2 = [(i+1, antigen[i], importance2[i]) 
                    for i in interface2_pos if i < len(antigen)]
        residues2.sort(key=lambda x: x[2], reverse=True)
        
        for pos, res, score in residues2[:10]:
            print(f"  {res}{pos}: {score:.3f}")

if __name__ == '__main__':
    main()