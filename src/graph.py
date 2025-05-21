import numpy as np
import torch

keypoints = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 
    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 
    'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]
edges = [
    ('nose', 'left_eye'), ('nose', 'right_eye'), 
    ('left_eye', 'left_ear'), ('right_eye', 'right_ear'),
    ('left_ear', 'left_shoulder'), ('right_ear', 'right_shoulder'),
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
    ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
    ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'), 
    ('left_hip', 'right_hip'),
    ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'), 
    ('right_hip', 'right_knee'), ('right_knee', 'right_ankle')
]

class Graph():
    def __init__(self, num_node=17, max_hop=3, dilation=1):
        self.num_node = num_node
        self.max_hop = max_hop
        self.dilation = dilation

        # get edges
        # self.num_node, self.edge, self.connect_joint, self.parts = self._get_edge()
        self.edge, self.connect_joint = self._get_edge()

        # get adjacency matrix
        self.A = self._get_adjacency()

    def __str__(self):
        return self.A

    def _get_edge(self):
        
        # keypoints = {
        #     0: "nose",
        #     1: "left_eye",
        #     2: "right_eye",
        #     3: "left_ear",
        #     4: "right_ear",
        #     5: "left_shoulder",
        #     6: "right_shoulder",
        #     7: "left_elbow",
        #     8: "right_elbow",
        #     9: "left_wrist",
        #     10: "right_wrist",
        #     11: "left_hip",
        #     12: "right_hip",
        #     13: "left_knee",
        #     14: "right_knee",
        #     15: "left_ankle",
        #     16: "right_ankle"
        # }
        self_link = [(i, i) for i in range(self.num_node)]
        neighbor_link = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6),
                         (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
                         (11, 13), (13, 15), (12, 14), (14, 16)]
        # self.center = 0
        connect_joint = np.array([5,0,0,1,2,0,0,5,6,7,8,5,6,11,12,13,14])
        # parts = [
        #     np.array([5, 7, 9]),       # left_arm
        #     np.array([6, 8, 10]),      # right_arm
        #     np.array([11, 13, 15]),    # left_leg
        #     np.array([12, 14, 16]),    # right_leg
        #     np.array([5, 6, 11, 12, 0, 1, 2, 3, 4]),  # torso + head
        # ]

        edge = self_link + neighbor_link
        # return edge, connect_joint, parts
        return edge, connect_joint

    def _get_hop_distance(self):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge:
            A[j, i] = 1
            A[i, j] = 1
        hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(self.max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def _get_adjacency(self):
        hop_dis = self._get_hop_distance()
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[hop_dis == hop] = 1
        normalize_adjacency = self._normalize_digraph(adjacency)
        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):
            A[i][hop_dis == hop] = normalize_adjacency[hop_dis == hop]

        # A = torch.tensor(A, dtype=torch.float32)
        
        return A

    def _normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-1)
        AD = np.dot(A, Dn)
        return AD
    
graph = Graph()
A = torch.tensor(graph.A, dtype=torch.float32)
connect_joint = graph.connect_joint

