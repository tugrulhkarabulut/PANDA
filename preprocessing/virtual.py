from collections import Counter
import itertools

import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree, to_undirected, subgraph


class AddVirtualNodes(BaseTransform):
    def __init__(
        self,
        num_selected_nodes: int,
        workers_per_node: int = 2,
        replication_factor: int = 1,
        return_hetero_data: bool = False,
    ):
        self.num_selected_nodes = num_selected_nodes
        self.workers_per_node = workers_per_node
        self.replication_factor = min(replication_factor, workers_per_node)
        self.return_hetero_data = return_hetero_data
        super().__init__()

    def __call__(self, data: Data) -> Data:
        device = data.edge_index.device
        degrees = degree(data.edge_index[0], data.num_nodes)
        num_selected_nodes = min(self.num_selected_nodes, data.num_nodes)
        _, central_nodes = degrees.topk(num_selected_nodes)
        central_nodes_to_index = {
            cent_node: i for i, cent_node in enumerate(central_nodes.tolist())
        }
        new_nodes = torch.arange(
            data.num_nodes,
            data.num_nodes + num_selected_nodes * self.workers_per_node,
            device=device,
        )


        ## Distributing out edges
        central_edges_out_mask = torch.isin(data.edge_index[0], central_nodes)
        central_edges_out = data.edge_index[:, central_edges_out_mask]
        central_edges_out = torch.repeat_interleave(
            central_edges_out, repeats=self.replication_factor, dim=1
        )
        central_edges_source_indices = []
        central_edge_counter = Counter()
        for node in central_edges_out[0].tolist():
            local_index = central_edge_counter[node] % self.workers_per_node
            global_index = (
                data.num_nodes
                + central_nodes_to_index[node] * self.workers_per_node
                + local_index
            )
            central_edges_source_indices.append(global_index)
            central_edge_counter.update([node])

        central_edges_source_indices = torch.tensor(central_edges_source_indices)
        central_edges_out[0] = central_edges_source_indices

        # Distributing in edges
        central_edges_in_mask = torch.isin(data.edge_index[1], central_nodes)
        central_edges_in = data.edge_index[:, central_edges_in_mask]
        central_edges_in = torch.repeat_interleave(
            central_edges_in, repeats=self.replication_factor, dim=1
        )
        central_edges_dest_indices = []
        central_edge_counter = Counter()
        for node in central_edges_in[1].tolist():
            local_index = central_edge_counter[node] % self.workers_per_node
            global_index = (
                data.num_nodes
                + central_nodes_to_index[node] * self.workers_per_node
                + local_index
            )
            central_edges_dest_indices.append(global_index)
            central_edge_counter.update([node])

        central_edges_dest_indices = torch.tensor(central_edges_dest_indices)
        central_edges_in[1] = central_edges_dest_indices

        node_to_virt_edges = torch.tensor(
            [
                edge
                for virt_nodes in torch.split(new_nodes, self.workers_per_node)
                for edge in itertools.combinations(virt_nodes, 2)
            ],
            device=device,
        ).T
        node_to_virt_edges = to_undirected(node_to_virt_edges)


        central_edges_out_expanded = []
        for source, dest in central_edges_out.T.tolist():
            if dest in central_nodes_to_index:
                virt_index_start = (
                    data.num_nodes
                    + central_nodes_to_index[dest] * self.workers_per_node
                )
                for virt_index in range(virt_index_start, virt_index_start + self.workers_per_node):
                    central_edges_out_expanded.append([source, virt_index])
            else:
                central_edges_out_expanded.append([source, dest])
        central_edges_out_expanded = torch.tensor(central_edges_out_expanded, device=device).T

        central_edges_in_expanded = []
        for source, dest in central_edges_in.T.tolist():
            if source in central_nodes_to_index:
                virt_index_start = (
                    data.num_nodes
                    + central_nodes_to_index[source] * self.workers_per_node
                )
                for virt_index in range(virt_index_start, virt_index_start + self.workers_per_node):
                    central_edges_in_expanded.append([virt_index, dest])
            else:
                central_edges_in_expanded.append([source, dest])
        central_edges_in_expanded = torch.tensor(central_edges_in_expanded, device=device).T

        edge_types = torch.cat(
            [
                torch.zeros(data.edge_index.shape[1], device=device),
                torch.ones(central_edges_out_expanded.shape[1], device=device),
                torch.ones(central_edges_in_expanded.shape[1], device=device),
                torch.full((node_to_virt_edges.shape[1],), 2.0, device=device),
            ],
        )
        data.edge_index = torch.cat(
            [data.edge_index, central_edges_out_expanded, central_edges_in_expanded, node_to_virt_edges], dim=1
        )
        etype_to_str = {
            0: "orig",
            1: "centr",
            2: "virt",
        }

        virtual_mask = torch.zeros(
            data.num_nodes + num_selected_nodes * self.workers_per_node,
            dtype=torch.bool,
        )
        virtual_mask[data.num_nodes :] = True
        data.virtual_mask = virtual_mask

        data.num_nodes = data.num_nodes + num_selected_nodes * self.workers_per_node

        if "x" in data:
            new_nodes_x = data.x[central_nodes]
            new_nodes_x = torch.repeat_interleave(
                new_nodes_x, repeats=self.workers_per_node, dim=0
            )
            new_x = torch.cat([data.x, new_nodes_x], dim=0)
            data.x = new_x
        if "y" in data and data.y.ndim > 1:
            new_nodes_y = data.y[central_nodes]
            new_nodes_y = torch.repeat_interleave(
                new_nodes_y, repeats=self.workers_per_node
            )
            new_y = torch.cat([data.y, new_nodes_y], dim=0)
            data.y = new_y

        nodes_without_originals_mask = torch.ones(
            data.num_nodes, dtype=torch.bool, device=device
        )
        nodes_without_originals_mask[central_nodes] = False
        nodes_without_originals = torch.arange(data.num_nodes, device=device)[
            nodes_without_originals_mask
        ]
        edge_index, edge_types = subgraph(
            nodes_without_originals,
            data.edge_index,
            edge_types,
            relabel_nodes=True,
            num_nodes=data.num_nodes,
        )
        data.edge_index = edge_index

        if "x" in data:
            data.x = data.x[nodes_without_originals]

        if "y" in data and data.y.ndim > 1:
            data.y = data.y[nodes_without_originals]

        data.num_nodes = data.num_nodes - num_selected_nodes
        
        if self.return_hetero_data:
            hetero_data = HeteroData()
            hetero_data["node"].x = data.x
            hetero_data["node"].num_nodes = data.num_nodes
            hetero_data.y = data.y

            for e_typ in [0, 1, 2]:
                hetero_data[
                    "node", etype_to_str[e_typ], "node"
                ].edge_index = data.edge_index[:, edge_types == e_typ]

            return hetero_data

        
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_selected_nodes={self.num_selected_nodes}, workers_per_node={self.workers_per_node}, replication_factor={self.replication_factor})"
