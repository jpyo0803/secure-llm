
def TopologicalSort(graph):
    # I want lower number to be processed first if they are in the same order set
    order = []

    def dfs(u):
        visited[u] = True
        for v in graph[u][::-1]:
            if not visited[v]:
                dfs(v)
        order.append(u + 1)

    visited = [False] * len(graph)
    for u in range(len(graph)):
        if not visited[u]:
            dfs(u)

    order.reverse() 

    return order


f = open('tools/dependency_graph.txt', 'r')

graph = [[] for _ in range(14)]

for x in f:
    u, v = map(int, x.strip().split())
    graph[u - 1].append(v - 1)

order = TopologicalSort(graph)
print(order)
