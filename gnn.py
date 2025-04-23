"""
Hierarchical 1,2-GNN for HIV Classification inspired by the paper:
"Graph Neural Networks for Prediction of Fuel Ignition Quality"
Schweidtmann et al., 2020

Implements a hierarchical GNN with a 1-GNN pass followed by a 2-GNN pass.

- Implemented random oversampling for the training set to handle class imbalance.
- Removed weighted BCE loss calculation; using standard BCEWithLogitsLoss now.
- Initializes 2-node features based on mean of 1-GNN outputs,without f_iso(s).
- Added ReduceLROnPlateau learning rate scheduler.
- Added Accuracy metric calculation for train/val/test.
- Added Weight Decay to Adam optimizer.
- Added Early Stopping based on validation AUC.
- Default to run on FULL dataset (DEBUG_SUBSET_SIZE=None).
- Still uses ONLY GNN, without ChemBERTa embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform
from itertools import combinations
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
import time
import os
import os.path as osp 
import numpy as np
import traceback 


warnings.filterwarnings("ignore", category=UserWarning)

warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")


# --- Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET_NAME = 'HIV'
ROOT_DIR = 'data/MoleculeNet_Raw'
BATCH_SIZE = 8
INITIAL_LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-5      
EPOCHS = 100              
EARLY_STOPPING_PATIENCE = 15 # Stop if val AUC doesn't improve for this many epochs
GNN_HIDDEN_DIM = 64
GNN_LAYERS_1 = 5
GNN_LAYERS_2 = 5
MLP_HIDDEN_DIM_1 = 128
MLP_HIDDEN_DIM_2 = 64
SEED = 42
# Set to None to run on the full dataset by default
DEBUG_SUBSET_SIZE = None # Set to an integer (e.g., 20000) to run on a subset for debug
 
# LR Scheduler Params
LR_SCHEDULER_FACTOR = 0.8
LR_SCHEDULER_PATIENCE = 5

# --- Define Checkpoint Filename and Full Path for oversampling version ---
CHECKPOINT_FILENAME = f'best_model_hgnn_oversampled_{"subset" if DEBUG_SUBSET_SIZE else "full"}_wd_es_corrected_v6.pth'
CHECKPOINT_FULL_PATH = osp.join(ROOT_DIR, CHECKPOINT_FILENAME) 

print(f"Using device: {DEVICE}")
if DEBUG_SUBSET_SIZE:
    print(f"!!! RUNNING IN DEBUG MODE ON {DEBUG_SUBSET_SIZE} SAMPLES !!!")
else:
    print("--- RUNNING ON FULL DATASET ---")
print(f"Hyperparameters: HiddenDim={GNN_HIDDEN_DIM}, Layers={GNN_LAYERS_1}+{GNN_LAYERS_2}, InitLR={INITIAL_LEARNING_RATE}, WeightDecay={WEIGHT_DECAY}")
print(f"Batch Size: {BATCH_SIZE}, Max Epochs: {EPOCHS}, Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
# --- Indicate using Oversampling,hiv dataset is very imblance ---
print(f"Using Oversampling for Training Set. Checkpoint Path: {CHECKPOINT_FULL_PATH}")


# Set random seed for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


# --- 2-GNN data Transform ---
class TwoGNNTransform(BaseTransform):
    """ Constructs the 2-graph high order garph structure (node pairs, edges). """
    def __call__(self, data):
        num_nodes = data.num_nodes
        # Check if x exists before accessing num_node_features
        if not hasattr(data, 'x') or data.x is None:
              print(f"Warning: Data object missing 'x' attribute or 'x' is None. Skipping 2-GNN transform.") 
              data.edge_index_2 = torch.empty((2, 0), dtype=torch.long)
              data.node_pairs = torch.empty((0, 2), dtype=torch.long)
              data.num_nodes_2 = 0 # Assign 0 explicitly
              return data

        # Check if edge_index exists before proceeding
        if not hasattr(data, 'edge_index') or data.edge_index is None:
              print(f"Warning: Data object missing 'edge_index'. Skipping 2-GNN transform.") 
              data.edge_index_2 = torch.empty((2, 0), dtype=torch.long)
              data.node_pairs = torch.empty((0, 2), dtype=torch.long)
              data.num_nodes_2 = 0 # Assign 0 explicitly
              return data

        num_node_features = data.num_node_features
        if num_nodes < 2:
            data.edge_index_2 = torch.empty((2, 0), dtype=torch.long)
            data.node_pairs = torch.empty((0, 2), dtype=torch.long)
            data.num_nodes_2 = 0 # Assign 0 explicitly
            return data
        node_indices = list(range(num_nodes))
        # Use combinations to generate unique pairs
        node_pairs = list(combinations(node_indices, 2))
        num_nodes_2 = len(node_pairs)
        data.node_pairs = torch.tensor(node_pairs, dtype=torch.long)
        data.num_nodes_2 = num_nodes_2 # Store the count of pairs as a Python int

        # Build edges for the 2-graph (pairs share one node)
        edge_list_2 = []
        pair_indices = list(range(num_nodes_2))

        # Build adjacency list for the 1-graph
        adj = {i: set() for i in range(num_nodes)}
        # Check if edge_index exists and is not empty
        if data.edge_index is not None and data.edge_index.numel() > 0:
            for i, j in data.edge_index.t().tolist():
                # Check bounds just in case
                if i < num_nodes and j < num_nodes:
                     adj[i].add(j)
                     adj[j].add(i)
                # else:
                     # print(f"Warning: Invalid edge index ({i}, {j}) for num_nodes={num_nodes}")

        # Create edges between pairs that share a node
        for i, j in combinations(pair_indices, 2): # O(N**4)
              pair_i_nodes = set(node_pairs[i])
              pair_j_nodes = set(node_pairs[j])
              # Check if the pairs share exactly one node
              if len(pair_i_nodes.intersection(pair_j_nodes)) == 1:
                  edge_list_2.extend([[i, j], [j, i]]) # Add edges in both directions

        if edge_list_2:
            data.edge_index_2 = torch.tensor(edge_list_2, dtype=torch.long).t().contiguous()
        else:
            data.edge_index_2 = torch.empty((2, 0), dtype=torch.long)

        return data

    def __repr__(self):
        return f'{self.__class__.__name__}()'


# --- Custom Dataset Class for Transformed Subset/Fullset ---
class TransformedDataset(InMemoryDataset):
    def __init__(self, root, original_dataset, transform=None, pre_transform=None, pre_filter=None):
        self.original_dataset = original_dataset
        # Pass the TwoGNNTransform as the pre_transform
        super().__init__(root, transform, pre_transform=TwoGNNTransform(), pre_filter=pre_filter)
        # Load processed data
        try:
            self.data, self.slices = torch.load(self.processed_paths[0])
        except FileNotFoundError:
            print(f"Processed file not found at {self.processed_paths[0]}. Processing dataset...")
            self.process()
            self.data, self.slices = torch.load(self.processed_paths[0])
        except Exception as e:
              print(f"Error loading processed file: {e}. Attempting to re-process...")
              self.process()
              self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_file_names(self):
        # We don't download raw files directly here, they come from MoleculeNet
        return []

    @property
    def processed_file_names(self):
        # Filename depends on whether it's a subset
        if DEBUG_SUBSET_SIZE:
              return [f'transformed_subset_{DEBUG_SUBSET_SIZE}_data.pt']
        else:
              return ['transformed_full_data.pt']


    def download(self):
        # Data is assumed to be downloaded by MoleculeNet already
        pass

    def process(self):
        print(f"Applying transform to {len(self.original_dataset)} samples...")
        # Filter out data points without node features 'x' or 'edge_index' first
        data_list = [data for data in tqdm(self.original_dataset, desc="Filtering valid data")
                     if hasattr(data, 'x') and data.x is not None and hasattr(data, 'edge_index')]

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        processed_data_list = []
        if self.pre_transform is not None:
            # Apply the TwoGNNTransform
            for idx, data in enumerate(tqdm(data_list, desc="Applying Pre-Transform (2-GNN)")):
                try:
                    processed_data = self.pre_transform(data.clone()) # Clone to avoid modifying original
                    # Basic validation after transform
                    if hasattr(processed_data, 'num_nodes_2'): # Check if the transform added the attribute
                        processed_data_list.append(processed_data)
                    else:
                        print(f"Warning: Skipping data point {idx} - missing 'num_nodes_2' after transform.")
                except Exception as e:
                    print(f"\nError processing data point {idx} during pre_transform: {e}. Skipping.")
                    traceback.print_exc()
                    continue # Skip this data point
        else:
            processed_data_list = data_list # Should not happen as pre_transform is set

        if not processed_data_list:
              raise RuntimeError("No data points remaining after filtering and pre-processing!")

        # Collate and save
        data, slices = self.collate(processed_data_list)
        # Ensure directory exists before saving processed data
        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])
        print(f"Saved processed data to {self.processed_paths[0]}")


# --- GNN Layer Definitions ---
class GNNLayer(MessagePassing):
    """ Layer for 1-GNN (uses edge features) """
    def __init__(self, in_channels, out_channels, edge_feature_dim):
        super(GNNLayer, self).__init__(aggr='add') # Use 'add' aggregation
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Network to process edge features
        if edge_feature_dim is None or edge_feature_dim <= 0:
              # If no edge features, use identity (effectively ignores edge_attr in message)
              self.edge_feature_network = nn.Identity()
              self.uses_edge_features = False
              # Need a linear layer to transform message in update if no edge features
              self.lin_message = nn.Linear(in_channels, out_channels, bias=False)
        else:
              # Simple MLP for edge features -> weights for message passing
              # Based on paper idea, 1-gnn should use edge feature to generate message passing weight seperately 
              self.edge_feature_network = nn.Sequential(
                  nn.Linear(edge_feature_dim, 32),
                  nn.ReLU(),
                  nn.Linear(32, in_channels * out_channels) # Output: W_e(matrix) for each edge , later need reshape 
              )
              self.uses_edge_features = True
              # Linear layer applied to aggregated messages (which are already out_channels after bmm)
              self.lin_message = nn.Linear(out_channels, out_channels, bias=False)

        # GRU cell for node updates
        self.gru = nn.GRUCell(out_channels, out_channels)


    def forward(self, x, edge_index, edge_attr):
        # x: [N, in_channels], edge_index: [2, E], edge_attr: [E, edge_feature_dim]
        # Start message passing: message-->aggre-->update
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # x_j: [E, in_channels] (features of source nodes for each edge)
        # edge_attr: [E, edge_feature_dim]
        if not self.uses_edge_features or edge_attr is None or edge_attr.nelement() == 0:
              # If no edge features, message is just the neighbor node feature x_j.
              # The update step will handle the transformation to out_channels.
              return x_j # [E, in_channels]


        # Process edge features to get weights W_e
        edge_attr = edge_attr.float() # Ensure float type
        edge_weights = self.edge_feature_network(edge_attr).view(-1, self.in_channels, self.out_channels) # [E, in_c, out_c]

        # Calculate message m_j = W_e * x_j
        # edge_weights is [E, in_channels, out_channels] --> x_j needs to be [E, 1, in_channels] for batch matrix multiplication
        # result should be [E, 1, out_channels] -> squeeze to [E, out_channels]
        message = torch.bmm(x_j.unsqueeze(1), edge_weights).squeeze(1) # [E, out_channels]
        return message


    def update(self, aggr_out, x):
        # aggr_out: [N, ?] (aggregated messages for each node)
        #   - If uses_edge_features: [N, out_channels]
        #   - If not uses_edge_features: [N, in_channels]
        # x: [N, out_channels] (node features from previous layer/initial embedding)

        # Apply the linear transformation to ensure correct dimension before ReLU and GRU
        # self.lin_message handles both cases (in->out or out->out)
        transformed_message = F.relu(self.lin_message(aggr_out)) # [N, out_channels]

        # Update node features using GRU
        # GRU input: transformed_message, hidden_state: x
        new_hidden_state = self.gru(transformed_message, x) # [N, out_channels]
        return new_hidden_state


class GNNLayer2(MessagePassing):
    """ Layer for 2-GNN (no explicit edge features between node pairs),all edge share same matix in this case
        different from 1-gnn
       """
    def __init__(self, in_channels, out_channels):
        super(GNNLayer2, self).__init__(aggr='add') # Use 'add' aggregation
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Linear transformation for messages (applied to neighbor features x_j)
        self.lin_msg_transform = nn.Linear(in_channels, out_channels, bias=False)

        # GRU cell for node updates
        self.gru = nn.GRUCell(out_channels, out_channels)

        # Linear layer applied to aggregated messages before GRU
        self.lin_aggr = nn.Linear(out_channels, out_channels, bias=False)

    def forward(self, x, edge_index):
        # x: [N_2, in_channels], edge_index: [2, E_2] (N_2 is number of node pairs)
        # Start message passing
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        # x_j: [E_2, in_channels] (features of neighboring node pairs)
        # Apply linear transformation to neighbor features
        message = self.lin_msg_transform(x_j) # [E_2, out_channels]
        return message

    def update(self, aggr_out, x):
        # aggr_out: [N_2, out_channels] (aggregated messages for each node pair)
        # x: [N_2, out_channels] (node pair features from previous layer/initial embedding)

        # Transform aggregated messages
        transformed_aggr = F.relu(self.lin_aggr(aggr_out)) # [N_2, out_channels]

        # Update node pair features using GRU
        # GRU input: transformed_aggr, hidden_state: x
        new_hidden_state = self.gru(transformed_aggr, x) # [N_2, out_channels]
        return new_hidden_state


# --- Hierarchical GNN Model Definition ---
class HierarchicalGNN(torch.nn.Module):
    """ Hierarchical GNN model with 1-GNN and 2-GNN passes. """
    def __init__(self, num_node_features, num_edge_features, hidden_dim, n_layers_1, n_layers_2):
        super(HierarchicalGNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_edge_features = num_edge_features if num_edge_features is not None else 0 # Handle None case

        # 1-GNN Layers
        self.initial_embed_1 = nn.Linear(num_node_features, hidden_dim)
        self.layers_1 = nn.ModuleList([GNNLayer(hidden_dim, hidden_dim, self.num_edge_features) for _ in range(n_layers_1)])

        # 2-GNN Layers
        # Input to 2-GNN is derived from 1-GNN output, so input dim is hidden_dim
        self.initial_embed_2 = nn.Linear(hidden_dim, hidden_dim) # Embed the combined 1-GNN features
        self.layers_2 = nn.ModuleList([GNNLayer2(hidden_dim, hidden_dim) for _ in range(n_layers_2)])

    def _create_batch_2(self, data):
        """
        Creates the batch vector for the 2-graph nodes (node pairs).
        Calculates total_nodes_2 by iterating through the batch.
        """
        # data is a Batch object
        # Determine device from input data if possible
        device = data.x.device if hasattr(data, 'x') and data.x is not None else DEVICE

        if not hasattr(data, 'ptr'):
            # Handle single graph case or missing attributes
            if hasattr(data, 'num_nodes_2'):
                # Likely a single graph, batch vector is all zeros
                num_nodes_2_val = getattr(data, 'num_nodes_2', 0)
                # Ensure num_nodes_2_val is treated as an integer
                if isinstance(num_nodes_2_val, torch.Tensor):
                    num_nodes_2_val = num_nodes_2_val.item() if num_nodes_2_val.numel() == 1 else 0
                try:
                    num_nodes_2_int = int(num_nodes_2_val)
                except (ValueError, TypeError):
                    num_nodes_2_int = 0
                return torch.zeros(num_nodes_2_int, dtype=torch.long, device=device)
            # Cannot determine batch assignment
            # print("Warning: Could not create batch_2 due to missing 'ptr'.") # Reduce verbosity
            return None

        # --- Calculate total_nodes_2 and batch_2_list reliably ---
        batch_size = data.num_graphs
        batch_2_list = []
        total_nodes_2 = 0 # This will be a python int

        # Iterate through graphs in the batch using to_data_list()
        try:
            for i, single_graph_data in enumerate(data.to_data_list()):
                num_nodes_2_in_graph = getattr(single_graph_data, 'num_nodes_2', 0)
                # Ensure it's an integer
                if isinstance(num_nodes_2_in_graph, torch.Tensor):
                     num_nodes_2_in_graph = num_nodes_2_in_graph.item() if num_nodes_2_in_graph.numel() == 1 else 0
                try:
                    num_nodes_2_in_graph = int(num_nodes_2_in_graph)
                except (ValueError, TypeError):
                    # print(f"Warning: Could not convert num_nodes_2 {num_nodes_2_in_graph} to int for graph {i}. Assuming 0.") # Reduce verbosity
                    num_nodes_2_in_graph = 0

                count = max(0, num_nodes_2_in_graph)
                batch_2_list.extend([i] * count)
                total_nodes_2 += count # Accumulate the count

        except Exception as e:
            print(f"Error during batch_2 creation using to_data_list: {e}. Returning None.")
            traceback.print_exc()
            return None

        # --- Sanity check removed ---

        if total_nodes_2 == 0:
            # Return empty tensor if no 2-nodes across the batch
            return torch.empty(0, dtype=torch.long, device=device)

        if not batch_2_list: # Should be redundant if total_nodes_2 > 0
            # This case might occur if total_nodes_2 > 0 but the loop failed silently
            print("Warning: batch_2_list is empty despite calculated total_nodes_2 > 0.")
            return torch.empty(0, dtype=torch.long, device=device) # Return empty tensor

        return torch.tensor(batch_2_list, dtype=torch.long, device=device)


    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        # Safely get node_pairs and edge_index_2, handle if they don't exist
        node_pairs = getattr(data, 'node_pairs', None)
        edge_index_2 = getattr(data, 'edge_index_2', None)

        # --- 1-GNN Pass ---
        # Initial node embedding
        x_1 = F.relu(self.initial_embed_1(x.float())) # [N, hidden_dim]

        # Apply GNN layers
        for layer in self.layers_1:
            # Handle potentially missing edge_attr
            current_edge_attr = edge_attr
            if current_edge_attr is None and self.num_edge_features > 0:
                # Create zero edge features if expected but missing
                num_edges = edge_index.size(1)
                current_edge_attr = torch.zeros((num_edges, self.num_edge_features), dtype=x_1.dtype, device=x_1.device)
            elif current_edge_attr is not None and self.num_edge_features == 0:
                # Pass None if no edge features are expected by the layer
                current_edge_attr = None

            x_1 = layer(x_1, edge_index, current_edge_attr) # [N, hidden_dim]

        # Global pooling for 1-graph embedding
        # Handle case where batch might be None (single graph inference)
        current_batch = batch
        if current_batch is None:
             current_batch = torch.zeros(x_1.shape[0], dtype=torch.long, device=x_1.device)
        graph_embedding_1 = global_add_pool(x_1, current_batch) # [batch_size, hidden_dim]


        # --- 2-GNN Pass ---
        x_2_initial = None
        batch_2 = self._create_batch_2(data) # [N_pairs_batch]

        # Check if we have node pairs and a valid batch vector for them
        # Also check if node_pairs and batch_2 have matching dimensions if both exist
        can_run_2gnn = False
        if node_pairs is not None and node_pairs.numel() > 0 and batch_2 is not None:
             if batch_2.shape[0] == node_pairs.shape[0]:
                 # Also check if there are edges for the 2-GNN
                 if edge_index_2 is not None and edge_index_2.numel() > 0:
                      can_run_2gnn = True
                 # else: # No 2-GNN edges, cannot proceed
                      # print("Debug: Skipping 2-GNN pass - no edge_index_2.") # Reduce verbosity
             # else: # Mismatch in counts
                 # print(f"Debug: Skipping 2-GNN pass - Mismatch between node_pairs shape ({node_pairs.shape[0]}) and batch_2 shape ({batch_2.shape[0]}).") # Reduce verbosity
                 pass
        # else: # No node_pairs or batch_2 is None
             # print("Debug: Skipping 2-GNN pass - No node_pairs or batch_2 is None.") # Reduce verbosity
             pass


        if can_run_2gnn:
            try:
                # Get node indices adjusted for batching using ptr
                ptr = data.ptr # [batch_size + 1]
                # If ptr is None (e.g., single graph), create a dummy ptr
                if ptr is None:
                    ptr = torch.tensor([0, x_1.shape[0]], device=x_1.device)

                # Adjust node_pairs indices relative to the start of each graph in the batch
                # Ensure batch_2 is on the same device as ptr before indexing
                offset = ptr[batch_2.to(ptr.device)].unsqueeze(-1) # [N_pairs_batch, 1]
                adj_node_pairs = node_pairs + offset # [N_pairs_batch, 2] (absolute indices)

                # Check bounds before indexing x_1
                max_index_needed = adj_node_pairs.max()
                if max_index_needed >= x_1.shape[0]:
                     raise IndexError(f"Index out of bounds: Max index needed ({max_index_needed}) >= x_1 dim 0 ({x_1.shape[0]})")

                # Gather features from the 1-GNN output (x_1)
                features_v1 = x_1[adj_node_pairs[:, 0]] # [N_pairs_batch, hidden_dim]
                features_v2 = x_1[adj_node_pairs[:, 1]] # [N_pairs_batch, hidden_dim]

                # Initialize 2-node features (Mean as per paper/previous version)
                x_2_initial = (features_v1 + features_v2) / 2.0 # [N_pairs_batch, hidden_dim]

                # --- Run 2-GNN Layers ---
                # Initial embedding for 2-nodes
                x_2_embedded = F.relu(self.initial_embed_2(x_2_initial.float())) # [N_pairs_batch, hidden_dim]

                # Apply 2-GNN layers
                for layer in self.layers_2:
                    x_2_embedded = layer(x_2_embedded, edge_index_2) # [N_pairs_batch, hidden_dim]

                # Global pooling for 2-graph embedding
                # Ensure batch_2 is still valid before pooling
                if batch_2 is not None and batch_2.shape[0] == x_2_embedded.shape[0]:
                    graph_embedding_2 = global_add_pool(x_2_embedded, batch_2) # [batch_size, hidden_dim]
                else:
                    print(f"Warning: Mismatch between x_2_embedded shape ({x_2_embedded.shape[0]}) and batch_2 shape ({batch_2.shape[0] if batch_2 is not None else 'None'}) during pooling. Using zero embedding for 2-GNN.")
                    graph_embedding_2 = torch.zeros_like(graph_embedding_1)


            except IndexError as e:
                print(f"Error during 2-GNN pass (IndexError): {e}. Using zero embedding for 2-GNN.")
                print(f"x_1 shape: {x_1.shape}, node_pairs shape: {node_pairs.shape if node_pairs is not None else 'None'}, batch_2 shape: {batch_2.shape if batch_2 is not None else 'None'}")
                if 'adj_node_pairs' in locals():
                     print(f"adj_node_pairs shape: {adj_node_pairs.shape}, max index requested: {adj_node_pairs.max() if adj_node_pairs.numel() > 0 else 'N/A'}")
                graph_embedding_2 = torch.zeros_like(graph_embedding_1) # Fallback
                can_run_2gnn = False # Ensure we don't try to use results later
            except Exception as e:
                print(f"Error during 2-GNN pass: {e}. Using zero embedding for 2-GNN.")
                traceback.print_exc()
                graph_embedding_2 = torch.zeros_like(graph_embedding_1) # Fallback
                can_run_2gnn = False # Ensure we don't try to use results later

        # If 2-GNN couldn't run for any reason, use zero embedding
        if not can_run_2gnn:
            graph_embedding_2 = torch.zeros_like(graph_embedding_1) # [batch_size, hidden_dim]

        # Concatenate 1-GNN and 2-GNN graph embeddings
        final_graph_embedding = torch.cat([graph_embedding_1, graph_embedding_2], dim=-1) # [batch_size, hidden_dim * 2]
        return final_graph_embedding


# --- MLP Definition ---
class MLP(torch.nn.Module):
    """ MLP for classification, taking concatenated 1-GNN and 2-GNN embeddings. """
    def __init__(self, gnn_embed_dim_total, out_channels):
        super(MLP, self).__init__()
        in_channels = gnn_embed_dim_total # hidden_dim * 2
        self.mlp = torch.nn.Sequential(
            nn.Linear(in_channels, MLP_HIDDEN_DIM_1),
            nn.ReLU(),
            nn.BatchNorm1d(MLP_HIDDEN_DIM_1),
            nn.Dropout(0.4),
            nn.Linear(MLP_HIDDEN_DIM_1, MLP_HIDDEN_DIM_2),
            nn.ReLU(),
            nn.BatchNorm1d(MLP_HIDDEN_DIM_2),
            nn.Dropout(0.4),
            nn.Linear(MLP_HIDDEN_DIM_2, out_channels),
        )

    def forward(self, combined_embedding):
        # Add a check for batch size 1 in BatchNorm
        if combined_embedding.shape[0] == 1 and len(self.mlp) > 2 and isinstance(self.mlp[2], nn.BatchNorm1d):
             # Temporarily disable BatchNorm for batch size 1 if it exists
             # A better solution might be GroupNorm or LayerNorm if batch size 1 is frequent
             # Or handle prediction differently for single samples
             # print("Warning: Batch size is 1, skipping BatchNorm in MLP.") # Reduce verbosity
             temp_mlp = nn.Sequential(*[layer for i, layer in enumerate(self.mlp) if not isinstance(layer, nn.BatchNorm1d)])
             return temp_mlp(combined_embedding)
        return self.mlp(combined_embedding)


# --- Data Loading and Preparation ---
print("\nLoading original dataset...")
start_time = time.time()
try:
    # Load the full dataset first (needed for splits and oversampling base)
    # Use a different root to avoid conflicts with transformed data cache
    original_dataset_full = MoleculeNet(root='data/MoleculeNet_Raw', name=DATASET_NAME)
    print(f"Original full dataset loaded in {time.time() - start_time:.2f} seconds. Samples: {len(original_dataset_full)}")

    # Select subset if specified (applied AFTER getting splits)
    dataset_to_process = original_dataset_full # Start with the full dataset

    print(f"Processing/Loading transformed dataset (root: {ROOT_DIR})...")
    # Ensure root directory exists
    os.makedirs(ROOT_DIR, exist_ok=True)
    start_transform_time = time.time()
    # Process the dataset with the transform
    # Note: We process the full dataset here, subset selection happens later if needed
    dataset = TransformedDataset(root=ROOT_DIR, original_dataset=dataset_to_process)
    print(f"Transformed dataset ready in {time.time() - start_transform_time:.2f} seconds. Final size: {len(dataset)}")

    if len(dataset) == 0:
        print("Error: Processed dataset is empty after filtering/transform. Check data validity and transform logic.")
        exit()

    # Determine number of features (more robustly)
    num_node_features = dataset.num_node_features
    num_edge_features = None
    if hasattr(dataset, 'num_edge_features') and dataset.num_edge_features is not None:
         num_edge_features = dataset.num_edge_features
    else:
         # Check first element if dataset attr is missing
         if len(dataset) > 0 and hasattr(dataset[0], 'edge_attr') and dataset[0].edge_attr is not None:
               num_edge_features = dataset[0].num_edge_features
         else: # Fallback if first element also lacks edge_attr
               print("Warning: Could not reliably determine num_edge_features from dataset or first sample. Checking other samples...")
               for data_point in dataset:
                   if hasattr(data_point, 'edge_attr') and data_point.edge_attr is not None and data_point.edge_attr.numel() > 0:
                        num_edge_features = data_point.edge_attr.shape[1]
                        print(f"Determined num_edge_features = {num_edge_features} from sample.")
                        break
               if num_edge_features is None:
                   # Try determining from raw dataset if possible
                   if hasattr(original_dataset_full, 'num_edge_features'):
                       num_edge_features = original_dataset_full.num_edge_features
                       print(f"Determined num_edge_features = {num_edge_features} from original dataset.")
                   else:
                       num_edge_features = 3 # Default assumption if still not found
                       print(f"Warning: Still could not determine num_edge_features. Assuming default value: {num_edge_features}.")

    print(f"Dataset properties: num_node_features={num_node_features}, num_edge_features={num_edge_features}")


except Exception as e:
    print(f"Error loading/transforming dataset: {e}")
    traceback.print_exc()
    exit()


# --- Data Splitting and Oversampling ---
print("\nSplitting data and applying oversampling to training set...")
num_samples = len(dataset)

# Get standard splits if available, otherwise create manually
try:
    split_idx = dataset.get_idx_split()
    train_indices = split_idx['train'].numpy()
    val_indices = split_idx['valid'].numpy() # MoleculeNet uses 'valid'
    test_indices = split_idx['test'].numpy()
    print("Using standard MoleculeNet splits.")
except Exception as e:
    print(f"Could not get standard splits ({e}). Creating 80/10/10 random split.")
    if num_samples < 50:
        print(f"Warning: Dataset too small ({num_samples}) for train/val/test split. Using all data for all sets.")
        train_indices = np.arange(num_samples)
        val_indices = np.arange(num_samples)
        test_indices = np.arange(num_samples)
    else:
        train_cutoff = int(0.8 * num_samples)
        val_cutoff = int(0.9 * num_samples)
        indices = np.random.permutation(num_samples)
        train_indices = indices[:train_cutoff]
        val_indices = indices[train_cutoff:val_cutoff]
        test_indices = indices[val_cutoff:]

# Apply subset selection AFTER getting initial splits if DEBUG_SUBSET_SIZE is set
if DEBUG_SUBSET_SIZE:
     print(f"Applying DEBUG_SUBSET_SIZE={DEBUG_SUBSET_SIZE}...")
     # Reduce the size of each split proportionally, ensuring minimum size
     target_train = max(10, int(DEBUG_SUBSET_SIZE * (len(train_indices) / num_samples)))
     target_val = max(5, int(DEBUG_SUBSET_SIZE * (len(val_indices) / num_samples)))
     target_test = max(5, int(DEBUG_SUBSET_SIZE * (len(test_indices) / num_samples)))

     if len(train_indices) > target_train:
         train_indices = np.random.choice(train_indices, target_train, replace=False)
     if len(val_indices) > target_val:
         val_indices = np.random.choice(val_indices, target_val, replace=False)
     if len(test_indices) > target_test:
         test_indices = np.random.choice(test_indices, target_test, replace=False)
     print(f"Subset sizes: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")


# --- Oversampling Logic ---
print("Applying oversampling to training indices...")
# Get labels for the original training set
train_labels = dataset.y[train_indices].numpy().flatten() # Get labels corresponding to train_indices

# Find indices of positive (minority) and negative (majority) samples within the train_indices array
pos_indices_in_train = np.where(train_labels == 1)[0]
neg_indices_in_train = np.where(train_labels == 0)[0]

# Get the actual dataset indices for positive samples
positive_original_indices = train_indices[pos_indices_in_train]

num_positive = len(pos_indices_in_train)
num_negative = len(neg_indices_in_train)
print(f"Original train set: Positive={num_positive}, Negative={num_negative}")

oversampled_train_indices = list(train_indices) # Start with original indices

if num_positive > 0 and num_negative > num_positive:
    # Calculate how many times to replicate positive samples
    multiplier = int(num_negative / num_positive) - 1
    remainder = num_negative % num_positive

    print(f"Oversampling: Replicating positive samples ~{multiplier+1} times.")

    # Add full replications
    for _ in range(multiplier):
        oversampled_train_indices.extend(positive_original_indices)

    # Add remainder samples
    if remainder > 0 and len(positive_original_indices) > 0:
        remainder_indices = np.random.choice(positive_original_indices, remainder, replace=True) # Sample with replacement if remainder > num_positive
        oversampled_train_indices.extend(remainder_indices)

# Shuffle the final list of training indices
np.random.shuffle(oversampled_train_indices)
oversampled_train_indices = np.array(oversampled_train_indices) # Convert back to numpy array

print(f"Oversampled train set size: {len(oversampled_train_indices)}")

# Create datasets using the determined indices
train_dataset = dataset[torch.tensor(oversampled_train_indices, dtype=torch.long)]
val_dataset = dataset[torch.tensor(val_indices, dtype=torch.long)]
test_dataset = dataset[torch.tensor(test_indices, dtype=torch.long)]

print(f"\nFinal Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")


# --- Class Weights Calculation Removed ---
# print("\nCalculating class weights for training set...") - Removed
# ... weight calculation logic removed ...
# pos_weight_tensor = torch.tensor([pos_weight_value], device=DEVICE) - Removed


# --- Create DataLoaders ---
print("\nCreating DataLoaders...")
# Set num_workers=0 explicitly to avoid potential multiprocessing issues during debugging
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
print("DataLoaders created.")


# --- Model Initialization ---
print("\nInitializing Hierarchical GNN models...")
try:
    # Ensure num_node_features is valid
    if num_node_features is None or num_node_features <= 0:
          raise ValueError(f"Invalid num_node_features determined: {num_node_features}")

    model_gnn = HierarchicalGNN(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features, # Pass determined edge features
        hidden_dim=GNN_HIDDEN_DIM,
        n_layers_1=GNN_LAYERS_1,
        n_layers_2=GNN_LAYERS_2
    ).to(DEVICE)

    model_mlp = MLP(
        gnn_embed_dim_total=GNN_HIDDEN_DIM * 2, # Input is concatenation of 1-GNN and 2-GNN embeddings
        out_channels=1 # Binary classification
    ).to(DEVICE)
    print("Models initialized successfully.")
    # Print model parameter count
    total_params_gnn = sum(p.numel() for p in model_gnn.parameters() if p.requires_grad)
    total_params_mlp = sum(p.numel() for p in model_mlp.parameters() if p.requires_grad)
    print(f"GNN Parameters: {total_params_gnn:,}")
    print(f"MLP Parameters: {total_params_mlp:,}")
    print(f"Total Trainable Parameters: {total_params_gnn + total_params_mlp:,}")

except Exception as e:
    print(f"Error initializing models: {e}")
    traceback.print_exc()
    exit()


# --- Optimizer and Loss Function ---
optimizer = torch.optim.Adam(
    list(model_gnn.parameters()) + list(model_mlp.parameters()),
    lr=INITIAL_LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

# --- MODIFIED: Use standard BCE loss as oversampling handles imbalance ---
criterion = torch.nn.BCEWithLogitsLoss()
print(f"Using Adam optimizer with LR={INITIAL_LEARNING_RATE}, WeightDecay={WEIGHT_DECAY}")
print(f"Using standard BCEWithLogitsLoss (due to oversampling)")


# --- Learning Rate Scheduler ---
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='max', # Reduce LR when validation AUC stops increasing
    factor=LR_SCHEDULER_FACTOR,
    patience=LR_SCHEDULER_PATIENCE,
    verbose=True
)
print(f"Using ReduceLROnPlateau scheduler: factor={LR_SCHEDULER_FACTOR}, patience={LR_SCHEDULER_PATIENCE}")


# --- Training and Evaluation Loop ---
print(f"\n--- Starting Training (Max {EPOCHS} Epochs) ---")
best_val_auc = 0.0
epochs_no_improve = 0 # Counter for early stopping

train_start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    # --- Training Phase ---
    model_gnn.train()
    model_mlp.train()
    total_loss = 0
    all_train_preds, all_train_labels = [], []

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch_idx, batch in enumerate(progress_bar):
        try:
            batch = batch.to(DEVICE)
            # Ensure labels 'y' are present and correctly formatted
            if not hasattr(batch, 'y') or batch.y is None:
                # print(f"Warning: Skipping training batch {batch_idx} - missing 'y' attribute.") # Reduce verbosity
                continue
            y = batch.y.float().view(-1, 1) # Ensure [batch_size, 1] and float

            # Check for edge_attr consistency (handle missing edge_attr if needed by model)
            if not hasattr(batch, 'edge_attr'):
                batch.edge_attr = None # Explicitly set to None if missing

            optimizer.zero_grad()
            emb_combined = model_gnn(batch) # Get combined embedding [batch_size, hidden_dim*2]
            preds = model_mlp(emb_combined) # Get logits [batch_size, 1]

            # Check for NaN/Inf in predictions or labels
            if torch.isnan(preds).any() or torch.isinf(preds).any():
                print(f"\nWarning: NaN/Inf detected in predictions for training batch {batch_idx}. Skipping batch.")
                continue
            if torch.isnan(y).any() or torch.isinf(y).any():
                print(f"\nWarning: NaN/Inf detected in labels for training batch {batch_idx}. Skipping batch.")
                continue

            # Ensure preds and y have compatible shapes for loss calculation
            if preds.shape != y.shape:
                 print(f"\nWarning: Shape mismatch between preds ({preds.shape}) and y ({y.shape}) in training batch {batch_idx}. Skipping batch.")
                 continue

            loss = criterion(preds, y) # Calculate standard loss

            # Check for NaN/Inf in loss
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"\nWarning: NaN/Inf detected in loss for training batch {batch_idx}. Skipping backward pass.")
                continue

            loss.backward() # Backpropagate

            # Gradient clipping (optional but often helpful)
            torch.nn.utils.clip_grad_norm_(model_gnn.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(model_mlp.parameters(), max_norm=1.0)

            optimizer.step() # Update weights

            total_loss += loss.item() * batch.num_graphs # Accumulate loss scaled by batch size
            # Store predictions (probabilities) and labels for epoch metrics
            all_train_preds.append(torch.sigmoid(preds).detach().cpu())
            all_train_labels.append(y.detach().cpu())

            progress_bar.set_postfix({'loss': loss.item(), 'lr': optimizer.param_groups[0]['lr']})

        except Exception as e:
            print(f"\nError during training batch {batch_idx}: {e}")
            traceback.print_exc()
            # Print problematic batch details for inspection
            print("\nProblematic Training Batch Details:")
            print(f"Batch type: {type(batch)}")
            # Print attributes safely
            for attr in ['x', 'edge_index', 'edge_attr', 'y', 'batch', 'ptr', 'node_pairs', 'edge_index_2', 'num_nodes', 'num_edges', 'num_nodes_2']:
                 if hasattr(batch, attr):
                      val = getattr(batch, attr)
                      if isinstance(val, torch.Tensor):
                           print(f"  {attr}: shape={val.shape}, dtype={val.dtype}, device={val.device}")
                      elif isinstance(val, (list, tuple)) and len(val) > 10:
                           print(f"  {attr}: type={type(val)}, len={len(val)}") # Avoid printing huge lists
                      else:
                           print(f"  {attr}: {val}")
                 else:
                      print(f"  {attr}: Not present")
            # Decide whether to continue to next batch or stop
            # raise e # Re-raise the exception to stop execution
            print("Continuing to next batch...")
            continue # Continue to next batch

    # Calculate training metrics for the epoch
    train_acc = 0.0
    train_auc = 0.0
    # Ensure dataset length is > 0 to avoid division by zero
    dataset_len = len(train_loader.dataset) if train_loader and train_loader.dataset else 0
    avg_loss = total_loss / dataset_len if dataset_len > 0 else 0


    if all_train_preds and all_train_labels:
        try:
            all_train_preds_tensor = torch.cat(all_train_preds)
            all_train_labels_tensor = torch.cat(all_train_labels)

            # Ensure tensors are not empty and have matching sizes
            if all_train_preds_tensor.numel() > 0 and all_train_labels_tensor.numel() > 0 and all_train_preds_tensor.shape[0] == all_train_labels_tensor.shape[0]:
                train_preds_binary = (all_train_preds_tensor > 0.5).float()
                train_acc = accuracy_score(all_train_labels_tensor.numpy(), train_preds_binary.numpy())

                # Calculate AUC if both classes are present
                unique_labels = torch.unique(all_train_labels_tensor)
                if len(unique_labels) == 2:
                    # Ensure no NaN values in predictions before calculating AUC
                    if not torch.isnan(all_train_preds_tensor).any():
                         train_auc = roc_auc_score(all_train_labels_tensor.numpy(), all_train_preds_tensor.numpy())
                    else:
                         print("Warning: NaN values found in training predictions. Cannot calculate AUC.")
                         train_auc = 0.0
                else:
                    # This might happen in early epochs if only one class is sampled
                    # print("Warning: Only one class present in collected training batch labels. AUC is 0.") # Reduce verbosity
                    train_auc = 0.0 # Cannot compute AUC with only one class
            else:
                print("Warning: Empty or mismatched predictions/labels tensor during training metric calculation.")

        except Exception as e:
            print(f"Warning: Could not calculate train metrics: {e}")
            traceback.print_exc()

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch:03d} -> Train Loss: {avg_loss:.4f}, Train AUC: {train_auc:.4f}, Train Acc: {train_acc:.4f}, LR: {current_lr:.6f}")


    # --- Validation Phase ---
    model_gnn.eval()
    model_mlp.eval()
    all_val_preds, all_val_labels = [], []
    with torch.no_grad():
        progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch:03d} [Val]", leave=False)
        for batch_idx_val, batch in enumerate(progress_bar_val):
            try:
                batch = batch.to(DEVICE)
                if not hasattr(batch, 'y') or batch.y is None:
                    # print(f"Warning: Skipping validation batch {batch_idx_val} - missing 'y' attribute.") # Reduce verbosity
                    continue # Skip if no labels
                y_val = batch.y.float().view(-1, 1)

                if not hasattr(batch, 'edge_attr'): batch.edge_attr = None

                emb_combined = model_gnn(batch)
                preds_logits = model_mlp(emb_combined)
                preds_probs = torch.sigmoid(preds_logits) # Get probabilities

                # Check for NaN/Inf before appending
                if not torch.isnan(preds_probs).any() and not torch.isinf(preds_probs).any():
                    all_val_preds.append(preds_probs.cpu())
                    all_val_labels.append(y_val.cpu())
                else:
                    print(f"\nWarning: NaN/Inf detected in validation predictions for batch {batch_idx_val}. Skipping batch.")

            except Exception as e:
                print(f"\nError during validation batch {batch_idx_val}: {e}")
                traceback.print_exc()
                # Print problematic batch details for inspection
                print("\nProblematic Validation Batch Details:")
                print(f"Batch type: {type(batch)}")
                # Print attributes safely
                for attr in ['x', 'edge_index', 'edge_attr', 'y', 'batch', 'ptr', 'node_pairs', 'edge_index_2', 'num_nodes', 'num_edges', 'num_nodes_2']:
                     if hasattr(batch, attr):
                          val = getattr(batch, attr)
                          if isinstance(val, torch.Tensor):
                               print(f"  {attr}: shape={val.shape}, dtype={val.dtype}, device={val.device}")
                          elif isinstance(val, (list, tuple)) and len(val) > 10:
                               print(f"  {attr}: type={type(val)}, len={len(val)}") # Avoid printing huge lists
                          else:
                               print(f"  {attr}: {val}")
                     else:
                          print(f"  {attr}: Not present")
                # raise e # Re-raise the exception to stop execution
                print("Continuing to next validation batch...")
                continue # Continue to next batch

    # Calculate validation metrics
    val_auc = 0.0
    val_acc = 0.0
    if not all_val_preds or not all_val_labels:
        print("Validation failed: No valid predictions/labels collected.")
    else:
        try:
            all_val_preds_tensor = torch.cat(all_val_preds)
            all_val_labels_tensor = torch.cat(all_val_labels)

            if all_val_preds_tensor.numel() > 0 and all_val_labels_tensor.numel() > 0 and all_val_preds_tensor.shape[0] == all_val_labels_tensor.shape[0]:
                val_preds_binary = (all_val_preds_tensor > 0.5).float()
                val_acc = accuracy_score(all_val_labels_tensor.numpy(), val_preds_binary.numpy())

                # Calculate AUC if both classes are present
                unique_labels_val = torch.unique(all_val_labels_tensor)
                if len(unique_labels_val) == 2:
                     # Ensure no NaN values in predictions before calculating AUC
                    if not torch.isnan(all_val_preds_tensor).any():
                        val_auc = roc_auc_score(all_val_labels_tensor.numpy(), all_val_preds_tensor.numpy())
                    else:
                         print("Warning: NaN values found in validation predictions. Cannot calculate AUC.")
                         val_auc = 0.0
                else:
                    val_auc = 0.0 # Cannot compute AUC with only one class
                    print("Warning: Only one class present in validation set. AUC is 0.")
            else:
                print("Warning: Empty or mismatched predictions/labels tensor during validation metric calculation.")

        except Exception as e:
            print(f"Error calculating Validation Metrics: {e}")
            traceback.print_exc()
            val_auc = 0.0
            val_acc = 0.0

    epoch_duration = time.time() - epoch_start_time
    print(f"Epoch {epoch:03d} -> Val AUC: {val_auc:.4f}, Val Acc: {val_acc:.4f} (Duration: {epoch_duration:.2f}s)")

    # --- LR Scheduler Step ---
    # Step scheduler only if val_auc is not NaN
    if not np.isnan(val_auc):
        scheduler.step(val_auc)
    else:
        print("Skipping LR scheduler step due to NaN validation AUC.")

    # --- Check for Improvement and Early Stopping ---
    # Only update best model if val_auc is not NaN and improves
    if not np.isnan(val_auc) and val_auc > best_val_auc:
        best_val_auc = val_auc
        epochs_no_improve = 0 # Reset counter
        # Save the best model checkpoint
        try:
              # Ensure directory exists using the full path
              checkpoint_dir = osp.dirname(CHECKPOINT_FULL_PATH)
              os.makedirs(checkpoint_dir, exist_ok=True)
              # Save using the full path
              torch.save({
                  'epoch': epoch,
                  'model_gnn_state_dict': model_gnn.state_dict(),
                  'model_mlp_state_dict': model_mlp.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'val_auc': best_val_auc,
                  'train_loss': avg_loss, # Include last train loss for info
                  'num_node_features': num_node_features, # Save features info
                  'num_edge_features': num_edge_features, # Save features info
              }, CHECKPOINT_FULL_PATH)
              print(f"*** New best model saved with Val AUC: {best_val_auc:.4f} to {CHECKPOINT_FULL_PATH} ***\n")
        except Exception as e:
              print(f"Error saving checkpoint: {e}")
              traceback.print_exc()

    else:
        epochs_no_improve += 1
        print(f"Epochs without val_auc improvement: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}\n")

    # Early stopping check
    if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
        print(f"\n--- Early stopping triggered after {epoch} epochs. Best Val AUC: {best_val_auc:.4f} ---")
        break # Exit the training loop

# --- End of Training Loop ---
training_duration = time.time() - train_start_time
print(f"\n--- Training finished in {training_duration:.2f} seconds ---")


# --- Test Set Evaluation ---
print(f"\n--- Evaluating Best Model on Test Set ({CHECKPOINT_FULL_PATH}) ---") # Use full path
# Load the best checkpoint
# Check existence using the full path
if not osp.exists(CHECKPOINT_FULL_PATH):
     print(f"Error: No best model checkpoint found at '{CHECKPOINT_FULL_PATH}'. Cannot evaluate on test set.") # Use full path
else:
    try:
        # Load using the full path
        checkpoint = torch.load(CHECKPOINT_FULL_PATH, map_location=DEVICE)

        # Check if feature dimensions match before loading state dict
        saved_node_features = checkpoint.get('num_node_features', -1)
        saved_edge_features = checkpoint.get('num_edge_features', -2) # Use different default to distinguish
        if saved_node_features == -1: # Check if feature info was saved
             print("Warning: Checkpoint does not contain feature dimension info. Assuming current dimensions match.")
             num_node_features_load = num_node_features
             num_edge_features_load = num_edge_features
        elif saved_node_features != num_node_features or saved_edge_features != num_edge_features:
              print(f"Warning: Feature dimension mismatch! Model trained with node={saved_node_features}, edge={saved_edge_features}. Current data has node={num_node_features}, edge={num_edge_features}.")
              # Re-initialize model based on SAVED dimensions for loading state_dict
              print("Re-initializing model with SAVED feature dimensions for loading state_dict...")
              num_node_features_load = saved_node_features
              num_edge_features_load = saved_edge_features if saved_edge_features != -2 else None # Handle default case
        else:
              num_node_features_load = num_node_features
              num_edge_features_load = num_edge_features


        # Re-initialize models using the determined feature dimensions for loading
        model_gnn_test = HierarchicalGNN(
            num_node_features=num_node_features_load,
            num_edge_features=num_edge_features_load,
            hidden_dim=GNN_HIDDEN_DIM, n_layers_1=GNN_LAYERS_1, n_layers_2=GNN_LAYERS_2
        ).to(DEVICE)
        model_mlp_test = MLP(gnn_embed_dim_total=GNN_HIDDEN_DIM * 2, out_channels=1).to(DEVICE)

        model_gnn_test.load_state_dict(checkpoint['model_gnn_state_dict'])
        model_mlp_test.load_state_dict(checkpoint['model_mlp_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']} with Val AUC {checkpoint.get('val_auc', 'N/A'):.4f}") # Use .get for safety

        model_gnn_test.eval()
        model_mlp_test.eval()
        all_test_preds, all_test_labels = [], []
        with torch.no_grad():
            progress_bar_test = tqdm(test_loader, desc="[Test]", leave=False)
            for batch_idx_test, batch in enumerate(progress_bar_test):
                try:
                    batch = batch.to(DEVICE)
                    if not hasattr(batch, 'y') or batch.y is None:
                        # print(f"Warning: Skipping test batch {batch_idx_test} - missing 'y' attribute.") # Reduce verbosity
                        continue
                    y_test = batch.y.float().view(-1, 1)

                    if not hasattr(batch, 'edge_attr'): batch.edge_attr = None

                    # IMPORTANT: Handle potential feature mismatch during inference
                    if saved_node_features != -1 and (saved_node_features != num_node_features or saved_edge_features != num_edge_features):
                         print(f"Warning: Evaluating test batch {batch_idx_test} using model trained on different features. Results might be invalid.")
                         # Ideally, you would handle this, e.g., by adding/removing feature columns
                         # or stopping execution. For now, just warn and proceed.

                    emb_combined = model_gnn_test(batch) # Use test model
                    preds_logits = model_mlp_test(emb_combined) # Use test model
                    preds_probs = torch.sigmoid(preds_logits)

                    if not torch.isnan(preds_probs).any() and not torch.isinf(preds_probs).any():
                        all_test_preds.append(preds_probs.cpu())
                        all_test_labels.append(y_test.cpu())
                    else:
                        print(f"\nWarning: NaN/Inf detected in test predictions for batch {batch_idx_test}. Skipping batch.")

                except Exception as e:
                    print(f"\nError during test batch {batch_idx_test}: {e}")
                    traceback.print_exc()
                    # Print problematic batch details for inspection
                    print("\nProblematic Test Batch Details:")
                    print(f"Batch type: {type(batch)}")
                    # Print attributes safely
                    for attr in ['x', 'edge_index', 'edge_attr', 'y', 'batch', 'ptr', 'node_pairs', 'edge_index_2', 'num_nodes', 'num_edges', 'num_nodes_2']:
                         if hasattr(batch, attr):
                              val = getattr(batch, attr)
                              if isinstance(val, torch.Tensor):
                                   print(f"  {attr}: shape={val.shape}, dtype={val.dtype}, device={val.device}")
                              elif isinstance(val, (list, tuple)) and len(val) > 10:
                                   print(f"  {attr}: type={type(val)}, len={len(val)}") # Avoid printing huge lists
                              else:
                                   print(f"  {attr}: {val}")
                         else:
                              print(f"  {attr}: Not present")
                    # raise e # Re-raise the exception to stop execution
                    print("Continuing to next test batch...")
                    continue

        # Calculate test metrics
        test_auc = 0.0
        test_acc = 0.0
        if not all_test_preds or not all_test_labels:
            print("Test evaluation failed: No valid predictions/labels collected.")
        else:
            try:
                all_test_preds_tensor = torch.cat(all_test_preds)
                all_test_labels_tensor = torch.cat(all_test_labels)

                if all_test_preds_tensor.numel() > 0 and all_test_labels_tensor.numel() > 0 and all_test_preds_tensor.shape[0] == all_test_labels_tensor.shape[0]:
                    test_preds_binary = (all_test_preds_tensor > 0.5).float()
                    test_acc = accuracy_score(all_test_labels_tensor.numpy(), test_preds_binary.numpy())

                    # Calculate AUC if both classes are present
                    unique_labels_test = torch.unique(all_test_labels_tensor)
                    if len(unique_labels_test) == 2:
                        # Ensure no NaN values in predictions before calculating AUC
                        if not torch.isnan(all_test_preds_tensor).any():
                            test_auc = roc_auc_score(all_test_labels_tensor.numpy(), all_test_preds_tensor.numpy())
                        else:
                            print("Warning: NaN values found in test predictions. Cannot calculate AUC.")
                            test_auc = 0.0
                    else:
                        test_auc = 0.0
                        print("Warning: Only one class present in test set. AUC is 0.")
                else:
                    print("Warning: Empty or mismatched predictions/labels tensor during test metric calculation.")

            except Exception as e:
                print(f"Error calculating Test Metrics: {e}")
                traceback.print_exc()
                test_auc = 0.0
                test_acc = 0.0

        print(f"\n--- Final Test Results ---")
        print(f"Test AUC: {test_auc:.4f}")
        print(f"Test Acc: {test_acc:.4f}")

    except FileNotFoundError: # Catch specific error if checkpoint doesn't exist (already handled above, but good practice)
         print(f"Error: No best model checkpoint found at '{CHECKPOINT_FULL_PATH}'. Cannot evaluate on test set.") # Use full path
    except Exception as e:
        print(f"An error occurred during test evaluation: {e}")
        traceback.print_exc()

print("\n--- Script finished ---")
