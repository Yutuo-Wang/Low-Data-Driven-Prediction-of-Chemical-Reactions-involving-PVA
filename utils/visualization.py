import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import os
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors
from rdkit.Chem.Draw import MolDraw2DCairo
import torch
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx

logger = logging.getLogger(__name__)

class Plotter:
    """
    Advanced plotting utilities for PVA-ReAct framework.
    
    This class provides comprehensive visualization capabilities
    for model performance, reaction analysis, and result presentation.
    """
    
    def __init__(self, style: str = 'default', figsize: Tuple[int, int] = (10, 8)):
        """
        Initialize plotter.
        
        Args:
            style: Matplotlib style ('default', 'seaborn', 'ggplot', 'dark')
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize
        self._set_style()
        
        # Color schemes
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'success': '#28A745',
            'warning': '#FFC107',
            'danger': '#DC3545',
            'light': '#F8F9FA',
            'dark': '#343A40'
        }
        
        logger.info("Plotter initialized")
    
    def _set_style(self):
        """Set matplotlib style."""
        if self.style == 'seaborn':
            plt.style.use('seaborn-v0_8-whitegrid')
            sns.set_palette("husl")
        elif self.style == 'ggplot':
            plt.style.use('ggplot')
        elif self.style == 'dark':
            plt.style.use('dark_background')
        else:
            plt.style.use('default')
        
        # Set better default parameters
        plt.rcParams['figure.figsize'] = self.figsize
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['axes.labelsize'] = 14
    
    def plot_training_curves(self, history: Dict[str, List], 
                           metrics: List[str] = None,
                           save_path: str = None,
                           show: bool = True) -> plt.Figure:
        """
        Plot training and validation curves.
        
        Args:
            history: Training history dictionary
            metrics: List of metrics to plot
            save_path: Path to save figure
            show: Whether to display figure
            
        Returns:
            fig: Matplotlib figure
        """
        if metrics is None:
            metrics = ['loss', 'accuracy', 'r2']
        
        # Determine number of subplots
        n_metrics = len([m for m in metrics if m in history])
        if n_metrics == 0:
            logger.warning("No valid metrics found in history")
            return None
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]
        
        plot_idx = 0
        epochs = list(range(1, len(history.get('train_loss', [])) + 1))
        
        for metric in metrics:
            if metric not in history:
                continue
                
            ax = axes[plot_idx]
            
            # Plot training and validation curves
            train_key = f'train_{metric}'
            val_key = f'val_{metric}'
            
            if train_key in history:
                ax.plot(epochs, history[train_key], 'b-', label='Train', linewidth=2, alpha=0.8)
            if val_key in history:
                ax.plot(epochs, history[val_key], 'r-', label='Validation', linewidth=2, alpha=0.8)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training curves saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_prediction_scatter(self, y_true: np.ndarray, y_pred: np.ndarray,
                              title: str = "Prediction vs True Values",
                              save_path: str = None,
                              show: bool = True) -> plt.Figure:
        """
        Create scatter plot of predictions vs true values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            save_path: Path to save figure
            show: Whether to display figure
            
        Returns:
            fig: Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Calculate metrics
        r2 = np.corrcoef(y_true, y_pred)[0, 1] ** 2
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        # Create scatter plot
        scatter = ax.scatter(y_true, y_pred, alpha=0.6, s=50, 
                           c=np.abs(y_true - y_pred), cmap='viridis')
        
        # Add perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        
        # Add metrics to plot
        textstr = f'R² = {r2:.3f}\nMAE = {mae:.3f}\nRMSE = {rmse:.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
        
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Absolute Error')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Scatter plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                      title: str = "Residual Analysis",
                      save_path: str = None,
                      show: bool = True) -> plt.Figure:
        """
        Plot residual analysis.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            save_path: Path to save figure
            show: Whether to display figure
            
        Returns:
            fig: Matplotlib figure
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Residuals vs predicted
        axes[0].scatter(y_pred, residuals, alpha=0.6, s=50)
        axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.8)
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs Predicted')
        axes[0].grid(True, alpha=0.3)
        
        # Residuals distribution
        axes[1].hist(residuals, bins=30, alpha=0.7, color=self.colors['primary'], edgecolor='black')
        axes[1].axvline(x=0, color='r', linestyle='--', alpha=0.8, label='Zero')
        axes[1].axvline(x=np.mean(residuals), color='orange', linestyle='-', 
                       alpha=0.8, label=f'Mean: {np.mean(residuals):.3f}')
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Residuals Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Residual plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_feature_importance(self, feature_names: List[str],
                              importance_scores: np.ndarray,
                              top_k: int = 20,
                              title: str = "Feature Importance",
                              save_path: str = None,
                              show: bool = True) -> plt.Figure:
        """
        Plot feature importance scores.
        
        Args:
            feature_names: List of feature names
            importance_scores: Importance scores for each feature
            top_k: Number of top features to show
            title: Plot title
            save_path: Path to save figure
            show: Whether to display figure
            
        Returns:
            fig: Matplotlib figure
        """
        # Sort features by importance
        indices = np.argsort(importance_scores)[-top_k:]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        y_pos = np.arange(len(indices))
        ax.barh(y_pos, importance_scores[indices], align='center', 
               color=self.colors['primary'], alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Importance Score')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            thresholds: List[float] = None,
                            title: str = "Confusion Matrix",
                            save_path: str = None,
                            show: bool = True) -> plt.Figure:
        """
        Plot confusion matrix for classification tasks.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            thresholds: Classification thresholds
            title: Plot title
            save_path: Path to save figure
            show: Whether to display figure
            
        Returns:
            fig: Matplotlib figure
        """
        if thresholds is None:
            thresholds = [30, 70]  # Low, medium, high yield
        
        # Convert to classification
        y_true_class = np.digitize(y_true, thresholds)
        y_pred_class = np.digitize(y_pred, thresholds)
        
        n_classes = len(thresholds) + 1
        class_names = [f'<{thresholds[0]}%'] + \
                     [f'{thresholds[i]}-{thresholds[i+1]}%' for i in range(len(thresholds)-1)] + \
                     [f'≥{thresholds[-1]}%']
        
        # Calculate confusion matrix
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for true, pred in zip(y_true_class, y_pred_class):
            cm[true, pred] += 1
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot confusion matrix
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        # Add text annotations
        thresh = cm.max() / 2
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(title)
        ax.set_xticks(np.arange(n_classes))
        ax.set_yticks(np.arange(n_classes))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def create_interactive_plot(self, data: pd.DataFrame,
                              x_col: str, y_col: str, 
                              color_col: str = None,
                              title: str = "Interactive Plot") -> go.Figure:
        """
        Create an interactive plot using Plotly.
        
        Args:
            data: DataFrame with plot data
            x_col: Column for x-axis
            y_col: Column for y-axis
            color_col: Column for color coding
            title: Plot title
            
        Returns:
            fig: Plotly figure
        """
        if color_col:
            fig = px.scatter(data, x=x_col, y=y_col, color=color_col,
                           title=title, hover_data=data.columns)
        else:
            fig = px.scatter(data, x=x_col, y=y_col, 
                           title=title, hover_data=data.columns)
        
        return fig


class ReactionVisualizer:
    """
    Visualization utilities for chemical reactions and molecules.
    """
    
    def __init__(self, image_size: Tuple[int, int] = (400, 400)):
        """
        Initialize reaction visualizer.
        
        Args:
            image_size: Default image size for molecule drawings
        """
        self.image_size = image_size
    
    def plot_molecule(self, smiles: str, 
                     title: str = None,
                     save_path: str = None,
                     show: bool = True) -> Optional[plt.Figure]:
        """
        Plot a molecule from SMILES string.
        
        Args:
            smiles: SMILES string
            title: Plot title
            save_path: Path to save image
            show: Whether to display image
            
        Returns:
            fig: Matplotlib figure (if show=False)
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                return None
            
            # Generate 2D coordinates
            AllChem.Compute2DCoords(mol)
            
            # Create drawing
            drawer = MolDraw2DCairo(self.image_size[0], self.image_size[1])
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            
            # Convert to image
            from PIL import Image
            import io
            
            img_data = drawer.GetDrawingText()
            img = Image.open(io.BytesIO(img_data))
            
            if show:
                plt.figure(figsize=(8, 8))
                plt.imshow(img)
                plt.axis('off')
                if title:
                    plt.title(title)
                plt.show()
            else:
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(img)
                ax.axis('off')
                if title:
                    ax.set_title(title)
                
                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    logger.info(f"Molecule image saved to {save_path}")
                
                return fig
            
            if save_path:
                img.save(save_path)
                logger.info(f"Molecule image saved to {save_path}")
                
        except Exception as e:
            logger.error(f"Failed to plot molecule {smiles}: {e}")
            return None
    
    def plot_reaction(self, reactant_smiles: str, product_smiles: str,
                     title: str = "Chemical Reaction",
                     save_path: str = None,
                     show: bool = True) -> Optional[plt.Figure]:
        """
        Plot a chemical reaction showing reactants and products.
        
        Args:
            reactant_smiles: Reactant SMILES
            product_smiles: Product SMILES
            title: Plot title
            save_path: Path to save image
            show: Whether to display image
            
        Returns:
            fig: Matplotlib figure (if show=False)
        """
        try:
            reactants = [Chem.MolFromSmiles(smi) for smi in reactant_smiles.split('.')]
            products = [Chem.MolFromSmiles(smi) for smi in product_smiles.split('.')]
            
            # Remove None values
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                logger.warning("Invalid reactants or products")
                return None
            
            # Generate 2D coordinates
            for mol in reactants + products:
                AllChem.Compute2DCoords(mol)
            
            # Create reaction visualization
            reaction_mols = reactants + products
            img = Draw.MolsToGridImage(reaction_mols, 
                                     molsPerRow=len(reactants),
                                     subImgSize=(300, 300),
                                     legends=['Reactant'] * len(reactants) + ['Product'] * len(products))
            
            if show:
                plt.figure(figsize=(12, 6))
                plt.imshow(img)
                plt.axis('off')
                plt.title(title)
                plt.show()
            else:
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(title)
                
                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    logger.info(f"Reaction image saved to {save_path}")
                
                return fig
            
            if save_path:
                img.save(save_path)
                logger.info(f"Reaction image saved to {save_path}")
                
        except Exception as e:
            logger.error(f"Failed to plot reaction: {e}")
            return None
    
    def plot_reaction_network(self, reactions: List[Dict[str, Any]],
                            save_path: str = None,
                            show: bool = True) -> Optional[plt.Figure]:
        """
        Plot a network of chemical reactions.
        
        Args:
            reactions: List of reaction dictionaries
            save_path: Path to save image
            show: Whether to display image
            
        Returns:
            fig: Matplotlib figure
        """
        try:
            # Create graph
            G = nx.DiGraph()
            
            # Add nodes and edges
            for i, reaction in enumerate(reactions):
                reactant = reaction.get('reactant_smiles', f'R{i}')
                product = reaction.get('product_smiles', f'P{i}')
                yield_val = reaction.get('yield', 0)
                
                G.add_node(reactant, type='reactant')
                G.add_node(product, type='product')
                G.add_edge(reactant, product, weight=yield_val, label=f'{yield_val:.1f}%')
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Layout
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # Draw nodes
            reactant_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'reactant']
            product_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'product']
            
            nx.draw_networkx_nodes(G, pos, nodelist=reactant_nodes, 
                                 node_color='lightblue', node_size=500, alpha=0.8)
            nx.draw_networkx_nodes(G, pos, nodelist=product_nodes,
                                 node_color='lightgreen', node_size=500, alpha=0.8)
            
            # Draw edges with weights
            edges = G.edges(data=True)
            edge_weights = [data['weight'] for _, _, data in edges]
            edge_colors = [plt.cm.RdYlGn(w/100) for w in edge_weights]  # Red to green based on yield
            
            nx.draw_networkx_edges(G, pos, edge_color=edge_colors, 
                                 width=2, alpha=0.7, arrows=True)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=8)
            
            # Draw edge labels (yields)
            edge_labels = {(u, v): data['label'] for u, v, data in edges}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
            
            ax.set_title('Chemical Reaction Network')
            ax.axis('off')
            
            # Add legend
            ax.plot([], [], 'o', color='lightblue', label='Reactants')
            ax.plot([], [], 'o', color='lightgreen', label='Products')
            ax.legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Reaction network saved to {save_path}")
            
            if show:
                plt.show()
            else:
                plt.close()
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to plot reaction network: {e}")
            return None


# Utility functions
def plot_training_history(history: Dict[str, List], 
                        save_path: str = None,
                        show: bool = True) -> plt.Figure:
    """
    Quick function to plot training history.
    
    Args:
        history: Training history dictionary
        save_path: Path to save figure
        show: Whether to display figure
        
    Returns:
        fig: Matplotlib figure
    """
    plotter = Plotter()
    return plotter.plot_training_curves(history, save_path=save_path, show=show)


def plot_molecule(smiles: str,
                 save_path: str = None,
                 show: bool = True) -> Optional[plt.Figure]:
    """
    Quick function to plot a molecule.
    
    Args:
        smiles: SMILES string
        save_path: Path to save image
        show: Whether to display image
        
    Returns:
        fig: Matplotlib figure (if show=False)
    """
    visualizer = ReactionVisualizer()
    return visualizer.plot_molecule(smiles, save_path=save_path, show=show)


def plot_reaction_network(reactions: List[Dict[str, Any]],
                         save_path: str = None,
                         show: bool = True) -> Optional[plt.Figure]:
    """
    Quick function to plot reaction network.
    
    Args:
        reactions: List of reaction dictionaries
        save_path: Path to save image
        show: Whether to display image
        
    Returns:
        fig: Matplotlib figure
    """
    visualizer = ReactionVisualizer()
    return visualizer.plot_reaction_network(reactions, save_path=save_path, show=show)


def create_performance_dashboard(metrics: Dict[str, Any],
                               predictions: Dict[str, np.ndarray],
                               save_path: str = None) -> go.Figure:
    """
    Create an interactive performance dashboard.
    
    Args:
        metrics: Performance metrics
        predictions: Prediction results
        save_path: Path to save dashboard
        
    Returns:
        fig: Plotly figure
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Prediction vs True', 'Residuals', 'Error Distribution', 'Metrics Summary'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    y_true = predictions.get('yield_true', [])
    y_pred = predictions.get('yield_pred', [])
    
    if len(y_true) > 0 and len(y_pred) > 0:
        # Scatter plot
        fig.add_trace(
            go.Scatter(x=y_true, y=y_pred, mode='markers', name='Predictions',
                      marker=dict(color='blue', opacity=0.6)),
            row=1, col=1
        )
        
        # Perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                      mode='lines', name='Perfect', line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        
        # Residuals
        residuals = np.array(y_true) - np.array(y_pred)
        fig.add_trace(
            go.Scatter(x=y_pred, y=residuals, mode='markers', name='Residuals',
                      marker=dict(color='orange', opacity=0.6)),
            row=1, col=2
        )
        
        # Error distribution
        absolute_errors = np.abs(residuals)
        fig.add_trace(
            go.Histogram(x=absolute_errors, name='Error Distribution',
                        marker_color='green', opacity=0.7),
            row=2, col=1
        )
    
    # Metrics summary
    metric_names = list(metrics.keys())
    metric_values = [metrics[name] for name in metric_names]
    
    fig.add_trace(
        go.Bar(x=metric_names, y=metric_values, name='Metrics',
              marker_color='purple'),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(height=800, title_text="Model Performance Dashboard")
    
    if save_path:
        fig.write_html(save_path)
        logger.info(f"Performance dashboard saved to {save_path}")
    
    return fig


# Example usage and testing
if __name__ == "__main__":
    print("Testing visualization utilities...")
    
    try:
        # Test Plotter
        plotter = Plotter()
        
        # Create sample data
        sample_history = {
            'train_loss': [1.0, 0.8, 0.6, 0.4, 0.3],
            'val_loss': [1.1, 0.9, 0.7, 0.5, 0.4],
            'train_r2': [0.5, 0.7, 0.8, 0.85, 0.9],
            'val_r2': [0.4, 0.6, 0.75, 0.8, 0.85]
        }
        
        y_true = np.random.uniform(0, 100, 100)
        y_pred = y_true + np.random.normal(0, 5, 100)
        
        # Test plotting functions
        plotter.plot_training_curves(sample_history, show=False)
        print("Training curves test passed")
        
        plotter.plot_prediction_scatter(y_true, y_pred, show=False)
        print("Scatter plot test passed")
        
        plotter.plot_residuals(y_true, y_pred, show=False)
        print("Residual plot test passed")
        
        # Test ReactionVisualizer
        visualizer = ReactionVisualizer()
        
        # Test molecule plotting
        visualizer.plot_molecule("CCO", show=False)  # Ethanol
        print("Molecule plotting test passed")
        
        # Test reaction plotting
        visualizer.plot_reaction("CCO", "CCOC(=O)O", show=False)
        print("Reaction plotting test passed")
        
        # Test reaction network
        sample_reactions = [
            {'reactant_smiles': 'CCO', 'product_smiles': 'CCOC(=O)O', 'yield': 85.5},
            {'reactant_smiles': 'c1ccccc1', 'product_smiles': 'c1ccccc1O', 'yield': 72.3}
        ]
        visualizer.plot_reaction_network(sample_reactions, show=False)
        print("Reaction network test passed")
        
        print("All visualization tests passed!")
        
    except Exception as e:
        print(f"Visualization test failed: {e}")
