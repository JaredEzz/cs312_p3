#!/usr/bin/python3

from CS312Graph import *
import time


class NetworkRoutingSolver:
    def __init__(self):
        pass

    def initializeNetwork(self, network):
        assert (type(network) == CS312Graph)
        self.network = network

    # Explanation of getShortestPath
    # Implementation of Dijkstra's algorithm to find shortest paths in a graph
    # parameters: destIndex - index of node to find shortest path of
    # return: map, two properties, cost - total length of the path, path - edges included in shortest path
    def getShortestPath(self, destIndex):
        self.dest = destIndex

        path_edges = []
        total_length = 0
        node = self.network.nodes[self.dest]

        # base case, source is the same as destination
        if self.network.nodes[self.dest] == self.source:
            return {'cost': total_length, 'path': path_edges}

        # loop until destination and source are connected
        while node.node_id != self.source:
            # Check previous node
            if self.results[node.node_id]['prev'] == -1:
                path_edges = []
                total_length = float("inf")
                break
            previousNode = self.network.nodes[self.results[node.node_id]['prev']]
            # Check nodes attached to previous node
            attachedNodes = previousNode.neighbors
            newEdge = None
            for attachedNode in attachedNodes:
                if attachedNode.dest.node_id == node.node_id:
                    newEdge = attachedNode
                    break
            # add found node
            path_edges.append((newEdge.src.loc, newEdge.dest.loc, '{:.0f}'.format(newEdge.length)))
            total_length += newEdge.length

            # update node for next iteration
            node = previousNode

        return {'cost': total_length, 'path': path_edges}

    # Explanation of computeShortestPaths
    # Computation of Dijkstra's algorithm, actual math performed here
    # parameters: srcIndex - index of source node to use in algorithm
    # return: elapsed time to execute the algorithm

    def computeShortestPaths(self, srcIndex, use_heap=False):
        self.source = srcIndex
        t1: float = time.time()

        # TODO: RUN DIJKSTRA'S TO DETERMINE SHORTEST PATHS.
        #       ALSO, STORE THE RESULTS FOR THE SUBSEQUENT
        #       CALL TO getShortestPath(dest_index)

        # begin Dijkstra's algorithm

        # Create priority queue initialized with infinite lengths
        priorityQueue = PriorityQueue(self.network, use_heap)

        # fill in 0 for the source, TODO -1 for source's previous node
        priorityQueue.updateCost(self.source, 0)

        # loop until all nodes are removed from the queue
        self.results = {}

        for node in self.network.nodes:
            self.results[node.node_id] = {'cost': float("inf"), 'prev': -1}

        # fill in 0 for source node cose
        self.results[self.source]['cost'] = 0

        while priorityQueue.isNotEmpty():
            # Find the node in the queue with the lowest cost
            lowestCostNode = priorityQueue.getLowestCost()
            # Find all the attached nodes
            attachedNodes = self.network.nodes[lowestCostNode['id']].neighbors
            for attachedNode in attachedNodes:
                neighborNode = self.results[attachedNode.dest.node_id]
                # Calculate cost of traveling from the node to the attached nodes
                newCost = lowestCostNode['cost'] + attachedNode.length
                if neighborNode['cost'] > newCost:
                    neighborNode['cost'] = newCost
                    neighborNode['prev'] = lowestCostNode['id']
                    priorityQueue.updateCost(attachedNode.dest.node_id, neighborNode['cost'])
        # end algorithm

        t2: float = time.time()
        return t2 - t1


class PriorityQueue:
    # Implementation for one of the following depending on UI input,
    # 1. Unsorted Array
    # 2. Min Heap
    # functions: getLowestCost - retrieves the node in the queue with the smallest distance
    # updateCost - updates the cost of a certain node

    # python equivalent of a constructor, array is default
    def __init__(self, network, useHeap=False):
        self.useHeap = useHeap
        self.nodeCount = len(network.nodes)

        # Initialize queue based on flag
        self.queue = MinHeap() if useHeap else {}
        # Add to the queue each node in the UI-generated graph
        for i in range(self.nodeCount):
            if useHeap:
                self.queue.insert(network.nodes[i].node_id, float("inf"))
            else:
                # Space Complexity O(n), an array value for every vertex, replaced when needed
                self.queue[network.nodes[i].node_id] = {'cost': float("inf")}

    # Time Complexity
    # Min Heap: O(1)
    # Unsorted Array: O(n) because each node must be visited to check length
    def getLowestCost(self):
        # Retrieve the next lowest cost node from the queue, aka has the highest priority
        if self.useHeap:
            return self.queue.getAndRemoveFirstNode()

        # Check each node, return the node with the lowest cost
        else:
            smallest_distance = float("inf")
            smallest_key = -1
            for m, n in self.queue.items():
                if self.queue[m]['cost'] < smallest_distance:
                    smallest_distance = self.queue[m]['cost']
                    smallest_key = m
            lowestCostNode = {'id': smallest_key, 'cost': smallest_distance}
            if smallest_key == -1:
                nextNode = self.queue.popitem()
                return {'id': nextNode[0], 'cost': nextNode[1]['cost']}
            # Remove the node from queue, O(1) because python method
            del self.queue[smallest_key]
            return lowestCostNode

    # Time Complexity
    # O(1) for both implementations
    def updateCost(self, nodeId, cost):
        if self.useHeap:
            self.queue.update(nodeId, cost)
        else:
            self.queue[nodeId]['cost'] = cost

    # Time Complexity
    # O(1) for both implementations
    def isNotEmpty(self):
        return self.queue.length() != 0 if self.useHeap else len(self.queue) != 0


# noinspection PyTypeChecker
# Space Complexity O(n), a node for every vertex, replaced when needed
class MinHeap:
    def __init__(self):
        self.heap = [-1]

    # Time complexity log(V) because percolate is log(V)
    def insert(self, nodeId, cost):
        newNode = {'id': nodeId, 'cost': cost}
        self.heap.append(newNode)
        self.percolateUp(len(self.heap) - 1, newNode)

    # Time complexity
    # log(V), worst case scenario is traversing the whole tree
    def percolateUp(self, index, newNode):
        while index != 1 and self.heap[index // 2]['cost'] > newNode['cost']:
            ancestor = self.heap[index // 2]
            self.heap[index // 2] = newNode
            self.heap[index] = ancestor
            index = index // 2

    # Time complexity
    # log(V), worst case scenario is traversing the whole tree
    def percolateDown(self, index, newNode):
        while 1:
            leaves = []  # refactor to leaves
            indices = []  # refactor to indices
            if (index * 2) <= (len(self.heap) - 1) and (self.heap[index * 2]['cost'] < newNode['cost']):
                indices.append(index * 2)
                leaves.append(self.heap[index * 2])
            if (index * 2 + 1) <= (len(self.heap) - 1) and (self.heap[index * 2 + 1]['cost'] < newNode['cost']):
                indices.append(index * 2 + 1)
                leaves.append(self.heap[index * 2 + 1])
            if len(leaves) == 0:
                break
            if len(leaves) == 1 or leaves[0]['cost'] < leaves[1]['cost']:
                leaf = leaves[0]
                leaf_index = indices[0]
            else:
                leaf = leaves[1]
                leaf_index = indices[1]

            self.heap[leaf_index] = self.heap[index]
            self.heap[index] = leaf
            index = leaf_index

    def length(self):
        return len(self.heap) - 1

    # Time complexity log(V) because percolate is log(V)
    def getAndRemoveFirstNode(self):
        firstNode = self.heap[1]
        newestNode = self.heap[self.length()]
        self.heap[1] = newestNode
        self.heap.pop()
        self.percolateDown(1, newestNode)
        return firstNode

    # Time Complexity
    # O(n) where n is number of nodes, because the heap is not complete
    def update(self, nodeId, cost):
        for i in range(1, len(self.heap)):
            if self.heap[i]['id'] == nodeId:
                self.heap[i]['cost'] = cost
                self.percolateUp(i, self.heap[i])
                break
