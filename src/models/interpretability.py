import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class AttentionExtractor:
    """Extract and analyze attention weights from MS-APPT model for biological interpretation."""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.attention_weights = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture attention weights."""
        self.hooks = []
        
        # Hook for self-attention
        def get_self_attention_hook(name):
            def hook(module, input, output):
                # Extract attention weights before dropout
                if hasattr(module, '_attention_weights'):
                    self.attention_weights[name] = module._attention_weights.detach().cpu()
            return hook
        
        # Hook for cross-attention
        def get_cross_attention_hook(name):
            def hook(module, input, output):
                if hasattr(module, '_attention_weights'):
                    self.attention_weights[name] = module._attention_weights.detach().cpu()
            return hook
        
        # Register hooks
        self.hooks.append(
            self.model.self_attention.register_forward_hook(
                get_self_attention_hook('self_attention')
            )
        )
        self.hooks.append(
            self.model.cross_attention.register_forward_hook(
                get_cross_attention_hook('cross_attention')
            )
        )
    
    def extract_attention_weights(self, protein1_embeddings: torch.Tensor,
                                protein2_embeddings: torch.Tensor,
                                protein1_sequences: List[str],
                                protein2_sequences: List[str]) -> Dict[str, torch.Tensor]:
        """Extract attention weights for a batch of protein pairs."""
        self.model.eval()
        self.attention_weights = {}
        
        with torch.no_grad():
            _ = self.model(protein1_embeddings, protein2_embeddings,
                         protein1_sequences, protein2_sequences)
        
        return self.attention_weights.copy()
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def analyze_binding_interface(self, cross_attention_weights: torch.Tensor,
                                protein1_sequence: str, protein2_sequence: str,
                                threshold: float = 0.1) -> Dict[str, np.ndarray]:
        """Analyze cross-attention to identify potential binding interfaces."""
        # Average over heads and batch
        if len(cross_attention_weights.shape) == 4:  # [batch, heads, seq1, seq2]
            avg_attention = cross_attention_weights[0].mean(dim=0).numpy()
        else:
            avg_attention = cross_attention_weights.numpy()
        
        # Identify high-attention regions
        protein1_importance = avg_attention.mean(axis=1)  # Average attention received
        protein2_importance = avg_attention.mean(axis=0)  # Average attention given
        
        # Find potential interface residues
        interface1_mask = protein1_importance > (protein1_importance.mean() + threshold * protein1_importance.std())
        interface2_mask = protein2_importance > (protein2_importance.mean() + threshold * protein2_importance.std())
        
        interface1_positions = np.where(interface1_mask)[0]
        interface2_positions = np.where(interface2_mask)[0]
        
        return {
            'protein1_importance': protein1_importance,
            'protein2_importance': protein2_importance,
            'interface1_positions': interface1_positions,
            'interface2_positions': interface2_positions,
            'attention_matrix': avg_attention
        }
    
    def get_residue_importance_scores(self, protein_sequence: str,
                                    importance_scores: np.ndarray) -> List[Tuple[int, str, float]]:
        """Get importance scores for each residue."""
        residue_scores = []
        seq_len = min(len(protein_sequence), len(importance_scores))
        
        for i in range(seq_len):
            residue_scores.append((i + 1, protein_sequence[i], float(importance_scores[i])))
        
        # Sort by importance
        residue_scores.sort(key=lambda x: x[2], reverse=True)
        return residue_scores
    
    def identify_hotspot_residues(self, residue_scores: List[Tuple[int, str, float]],
                                top_k: int = 10) -> List[Tuple[int, str, float]]:
        """Identify top-k hotspot residues based on attention scores."""
        return residue_scores[:top_k]


class BiologicalInterpreter:
    """Interpret MS-APPT predictions in biological context."""
    
    def __init__(self):
        self.hydrophobic_residues = set('AVILMFYW')
        self.charged_residues = set('DEKR')
        self.polar_residues = set('STNQ')
        self.aromatic_residues = set('FYW')
        self.small_residues = set('AGS')
        self.helix_breakers = set('PG')
    
    def analyze_sequence_properties(self, sequence: str, 
                                  interface_positions: np.ndarray) -> Dict[str, float]:
        """Analyze biological properties of predicted interface residues."""
        if len(interface_positions) == 0:
            return {}
        
        interface_residues = [sequence[i] for i in interface_positions if i < len(sequence)]
        total_interface = len(interface_residues)
        
        if total_interface == 0:
            return {}
        
        properties = {
            'hydrophobic_fraction': sum(1 for r in interface_residues if r in self.hydrophobic_residues) / total_interface,
            'charged_fraction': sum(1 for r in interface_residues if r in self.charged_residues) / total_interface,
            'polar_fraction': sum(1 for r in interface_residues if r in self.polar_residues) / total_interface,
            'aromatic_fraction': sum(1 for r in interface_residues if r in self.aromatic_residues) / total_interface,
            'small_fraction': sum(1 for r in interface_residues if r in self.small_residues) / total_interface,
            'helix_breaker_fraction': sum(1 for r in interface_residues if r in self.helix_breakers) / total_interface,
            'interface_size': total_interface,
            'interface_fraction': total_interface / len(sequence)
        }
        
        return properties
    
    def generate_interface_report(self, protein1_sequence: str, protein2_sequence: str,
                                interface_analysis: Dict, predicted_pkd: float) -> str:
        """Generate a biological interpretation report."""
        report = []
        report.append("=== MS-APPT Biological Interface Analysis ===\n")
        
        # Predicted affinity
        report.append(f"Predicted Binding Affinity:")
        report.append(f"  pKd: {predicted_pkd:.2f}")
        report.append(f"  Kd: {10**(-predicted_pkd):.2e} M")
        report.append(f"  Binding strength: {self._classify_binding_strength(predicted_pkd)}\n")
        
        # Protein 1 interface
        report.append("Protein 1 Interface:")
        interface1_positions = interface_analysis['interface1_positions']
        if len(interface1_positions) > 0:
            props1 = self.analyze_sequence_properties(protein1_sequence, interface1_positions)
            report.append(f"  Interface size: {props1['interface_size']} residues ({props1['interface_fraction']:.1%} of protein)")
            report.append(f"  Hydrophobic: {props1['hydrophobic_fraction']:.1%}")
            report.append(f"  Charged: {props1['charged_fraction']:.1%}")
            report.append(f"  Aromatic: {props1['aromatic_fraction']:.1%}")
            
            # Top hotspots
            importance1 = interface_analysis['protein1_importance']
            residue_scores1 = [(i+1, protein1_sequence[i], importance1[i]) 
                              for i in interface1_positions if i < len(protein1_sequence)]
            residue_scores1.sort(key=lambda x: x[2], reverse=True)
            
            report.append("  Top 5 hotspot residues:")
            for pos, res, score in residue_scores1[:5]:
                report.append(f"    {res}{pos}: {score:.3f}")
        else:
            report.append("  No significant interface detected")
        
        report.append("")
        
        # Protein 2 interface
        report.append("Protein 2 Interface:")
        interface2_positions = interface_analysis['interface2_positions']
        if len(interface2_positions) > 0:
            props2 = self.analyze_sequence_properties(protein2_sequence, interface2_positions)
            report.append(f"  Interface size: {props2['interface_size']} residues ({props2['interface_fraction']:.1%} of protein)")
            report.append(f"  Hydrophobic: {props2['hydrophobic_fraction']:.1%}")
            report.append(f"  Charged: {props2['charged_fraction']:.1%}")
            report.append(f"  Aromatic: {props2['aromatic_fraction']:.1%}")
            
            # Top hotspots
            importance2 = interface_analysis['protein2_importance']
            residue_scores2 = [(i+1, protein2_sequence[i], importance2[i]) 
                              for i in interface2_positions if i < len(protein2_sequence)]
            residue_scores2.sort(key=lambda x: x[2], reverse=True)
            
            report.append("  Top 5 hotspot residues:")
            for pos, res, score in residue_scores2[:5]:
                report.append(f"    {res}{pos}: {score:.3f}")
        else:
            report.append("  No significant interface detected")
        
        report.append("")
        
        # Interaction characteristics
        report.append("Predicted Interaction Characteristics:")
        if len(interface1_positions) > 0 and len(interface2_positions) > 0:
            # Complementarity analysis
            if props1['hydrophobic_fraction'] > 0.4 and props2['hydrophobic_fraction'] > 0.4:
                report.append("  - Likely hydrophobic-driven interaction")
            if props1['charged_fraction'] > 0.3 or props2['charged_fraction'] > 0.3:
                report.append("  - Significant electrostatic component")
            if props1['aromatic_fraction'] > 0.2 or props2['aromatic_fraction'] > 0.2:
                report.append("  - Potential π-π stacking interactions")
            
            # Interface size classification
            avg_interface_size = (props1['interface_size'] + props2['interface_size']) / 2
            if avg_interface_size < 10:
                report.append("  - Small interface (peptide-like)")
            elif avg_interface_size < 30:
                report.append("  - Medium interface (typical domain-domain)")
            else:
                report.append("  - Large interface (extensive contact)")
        
        return "\n".join(report)
    
    def _classify_binding_strength(self, pkd: float) -> str:
        """Classify binding strength based on pKd value."""
        if pkd < 5:
            return "Weak (transient interaction)"
        elif pkd < 8:
            return "Moderate (regulatory interaction)"
        elif pkd < 11:
            return "Strong (stable complex)"
        else:
            return "Very strong (constitutive complex)"


def visualize_attention_matrix(attention_matrix: np.ndarray,
                             protein1_sequence: str,
                             protein2_sequence: str,
                             save_path: Optional[str] = None,
                             figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
    """Visualize cross-attention matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(attention_matrix, cmap='YlOrRd', cbar=True, 
                square=False, ax=ax, cbar_kws={'label': 'Attention Weight'})
    
    # Set labels
    ax.set_xlabel('Protein 2 Position', fontsize=12)
    ax.set_ylabel('Protein 1 Position', fontsize=12)
    ax.set_title('Cross-Attention Matrix\n(Potential Binding Interface)', fontsize=14)
    
    # Add sequence ticks if sequences are short enough
    if len(protein1_sequence) <= 50:
        ax.set_yticks(np.arange(len(protein1_sequence)) + 0.5)
        ax.set_yticklabels([f"{res}{i+1}" for i, res in enumerate(protein1_sequence)], 
                          rotation=0, fontsize=8)
    
    if len(protein2_sequence) <= 50:
        ax.set_xticks(np.arange(len(protein2_sequence)) + 0.5)
        ax.set_xticklabels([f"{res}{i+1}" for i, res in enumerate(protein2_sequence)], 
                          rotation=90, fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved attention matrix visualization to {save_path}")
    
    return fig


def visualize_residue_importance(protein_sequence: str,
                               importance_scores: np.ndarray,
                               title: str = "Residue Importance Scores",
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (15, 6)) -> plt.Figure:
    """Visualize per-residue importance scores."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
    
    positions = np.arange(len(importance_scores))
    
    # Bar plot of importance scores
    bars = ax1.bar(positions, importance_scores, width=1.0)
    
    # Color by residue type
    colors = []
    hydrophobic = set('AVILMFYW')
    charged = set('DEKR')
    polar = set('STNQ')
    
    for i, res in enumerate(protein_sequence[:len(importance_scores)]):
        if res in hydrophobic:
            colors.append('orange')
        elif res in charged:
            colors.append('blue')
        elif res in polar:
            colors.append('green')
        else:
            colors.append('gray')
    
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax1.set_xlabel('Residue Position', fontsize=12)
    ax1.set_ylabel('Importance Score', fontsize=12)
    ax1.set_title(title, fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Sequence display
    if len(protein_sequence) <= 100:
        ax2.text(0.5, 0.5, protein_sequence, fontsize=10, 
                transform=ax2.transAxes, ha='center', va='center',
                family='monospace')
    else:
        ax2.text(0.5, 0.5, f"Sequence too long to display ({len(protein_sequence)} aa)", 
                fontsize=10, transform=ax2.transAxes, ha='center', va='center')
    
    ax2.set_xlim(0, len(importance_scores))
    ax2.axis('off')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='orange', label='Hydrophobic'),
        Patch(facecolor='blue', label='Charged'),
        Patch(facecolor='green', label='Polar'),
        Patch(facecolor='gray', label='Other')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved residue importance visualization to {save_path}")
    
    return fig