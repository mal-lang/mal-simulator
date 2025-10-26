from gymnasium.spaces import Box, Space
import numpy as np
from typing import Dict, Any, Tuple


class HeteroDataSpace(Space):
    """
    A space similar to PyTorch Geometric's HeteroData for handling heterogeneous graph data.
    
    This space can store different types of nodes and edges, each with their own attributes,
    facilitating the modeling of complex relationships in heterogeneous graph data.
    
    Example usage:
        space = HeteroDataSpace()
        space.add_node_type('asset', Box(0, 1, shape=(10,), dtype=np.float32))
        space.add_node_type('attack_step', Box(0, 1, shape=(5,), dtype=np.float32))
        space.add_edge_type(('asset', 'has', 'attack_step'), Box(0, 1, shape=(2,), dtype=np.int64))
    """
    
    def __init__(self):
        super().__init__()
        self._node_spaces: Dict[str, Space] = {}
        self._edge_spaces: Dict[Tuple[str, str, str], Space] = {}
        self._node_features: Dict[str, Any] = {}
        self._edge_features: Dict[Tuple[str, str, str], Any] = {}
    
    def add_node_type(self, node_type: str, space: Space) -> None:
        """
        Add a node type with its corresponding space.
        
        Args:
            node_type: Name of the node type (e.g., 'asset', 'attack_step')
            space: Gymnasium space defining the structure of this node type
        """
        self._node_spaces[node_type] = space
    
    def add_edge_type(self, edge_type: Tuple[str, str, str], space: Space) -> None:
        """
        Add an edge type with its corresponding space.
        
        Args:
            edge_type: Tuple of (source_type, relation, target_type)
            space: Gymnasium space defining the structure of this edge type
        """
        self._edge_spaces[edge_type] = space
    
    def set_node_features(self, node_type: str, features: Any) -> None:
        """
        Set features for a specific node type.
        
        Args:
            node_type: Name of the node type
            features: Feature data for this node type
        """
        if node_type not in self._node_spaces:
            raise ValueError(f"Node type '{node_type}' not found. Add it first with add_node_type().")
        self._node_features[node_type] = features
    
    def set_edge_features(self, edge_type: Tuple[str, str, str], features: Any) -> None:
        """
        Set features for a specific edge type.
        
        Args:
            edge_type: Tuple of (source_type, relation, target_type)
            features: Feature data for this edge type
        """
        if edge_type not in self._edge_spaces:
            raise ValueError(f"Edge type '{edge_type}' not found. Add it first with add_edge_type().")
        self._edge_features[edge_type] = features
    
    def get_node_features(self, node_type: str) -> Any:
        """Get features for a specific node type."""
        return self._node_features.get(node_type)
    
    def get_edge_features(self, edge_type: Tuple[str, str, str]) -> Any:
        """Get features for a specific edge type."""
        return self._edge_features.get(edge_type)
    
    def sample(self) -> Dict[str, Any]:
        """
        Sample a random observation from this space.
        
        Returns:
            Dictionary containing sampled data for all node and edge types
        """
        sample_data = {}
        
        # Sample node features
        for node_type, space in self._node_spaces.items():
            sample_data[node_type] = space.sample()
        
        # Sample edge features
        for edge_type, space in self._edge_spaces.items():
            sample_data[edge_type] = space.sample()
        
        return sample_data
    
    def contains(self, x: Any) -> bool:
        """
        Check if x is a valid observation in this space.
        
        Args:
            x: Observation to check
            
        Returns:
            True if x is valid, False otherwise
        """
        if not isinstance(x, dict):
            return False
        
        # Check node types
        for node_type, space in self._node_spaces.items():
            if node_type not in x:
                return False
            if not space.contains(x[node_type]):
                return False
        
        # Check edge types
        for edge_type, space in self._edge_spaces.items():
            if edge_type not in x:
                return False
            if not space.contains(x[edge_type]):
                return False
        
        return True
    
    @property
    def node_types(self) -> list[str]:
        """Get list of all node types."""
        return list(self._node_spaces.keys())
    
    @property
    def edge_types(self) -> list[Tuple[str, str, str]]:
        """Get list of all edge types."""
        return list(self._edge_spaces.keys())
    
    def __repr__(self) -> str:
        return f"HeteroDataSpace(node_types={self.node_types}, edge_types={self.edge_types})"


def create_mal_hetero_space(
    num_assets: int,
    num_attack_steps: int,
    num_asset_types: int,
    num_attack_step_types: int,
    asset_feature_dim: int = 10,
    attack_step_feature_dim: int = 5
) -> HeteroDataSpace:
    """
    Create a HeteroDataSpace specifically designed for MAL simulator data.
    
    Args:
        num_assets: Number of assets in the model
        num_attack_steps: Number of attack steps
        num_asset_types: Number of different asset types
        num_attack_step_types: Number of different attack step types
        asset_feature_dim: Dimension of asset features
        attack_step_feature_dim: Dimension of attack step features
        
    Returns:
        Configured HeteroDataSpace for MAL simulator
    """
    space = HeteroDataSpace()
    
    # Add node types
    space.add_node_type('asset', Box(
        low=0, high=1, 
        shape=(num_assets, asset_feature_dim), 
        dtype=np.float32
    ))
    
    space.add_node_type('attack_step', Box(
        low=0, high=1, 
        shape=(num_attack_steps, attack_step_feature_dim), 
        dtype=np.float32
    ))
    
    # Add edge types for attack graph relationships
    space.add_edge_type(('asset', 'has', 'attack_step'), Box(
        low=0, high=max(num_assets, num_attack_steps),
        shape=(2,), dtype=np.int64
    ))
    
    space.add_edge_type(('attack_step', 'leads_to', 'attack_step'), Box(
        low=0, high=num_attack_steps,
        shape=(2,), dtype=np.int64
    ))
    
    return space