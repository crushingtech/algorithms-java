package com.algorithms.graphs;

/**
 * Created by eugene on 2/21/16.
 */
public class ShortestPathMatrix {

    static final int V = 9;

    // Driver method
    public static void main (String[] args)
    {
        /* Let us create the example graph discussed above */
        int graph[][] = new int[][]{
                {0, 4, 0, 0, 0, 0, 0, 8, 0},
                {4, 0, 8, 0, 0, 0, 0, 11,0},
                {0, 8, 0, 7, 0, 4, 0, 0, 2},
                {0, 0, 7, 0, 9, 14,0, 0, 0},
                {0, 0, 0, 9, 0, 10,0, 0, 0},
                {0, 0, 4, 0, 10,0, 2, 0, 0},
                {0, 0, 0, 14,0, 2, 0, 1, 6},
                {8, 11,0, 0, 0, 0, 1, 0, 7},
                {0, 0, 2, 0, 0, 0, 6, 7, 0}
        };
        ShortestPathMatrix t = new ShortestPathMatrix();
        t.dijkstra(graph, 0);
    }

    int minDistance(int dist[], Boolean sptSet[]) {
        int min = Integer.MAX_VALUE;
        int minIndex = -1;
        for (int i = 0; i < V; i++) {
            if (sptSet[i] == false && dist[i] <= min) {
                min = dist[i];
                minIndex = i;
            }
        }
        return minIndex;
    }


    // A utility function to print the constructed distance array
    void printSolution(int dist[], int n)
    {
        System.out.println("Vertex   Distance from Source");
        for (int i = 0; i < V; i++)
            System.out.println(i+" \t\t "+dist[i]);
    }

    void dijkstra(int graph[][], int src){
        int dist[] = new int[V];
        Boolean[] sptSet = new Boolean[V];
        for (int i = 0; i < V; i++) {
            dist[i] = Integer.MAX_VALUE;
            sptSet[i] = false;
        }
        dist[src] = 0;
        for (int i = 0; i < V - 1; i++) {
            int u = minDistance(dist, sptSet);
            sptSet[u] = true;

            for (int v = 0; v < V; v++) {
                if(!sptSet[v] && graph[u][v]!=0 && dist[u]!=Integer.MAX_VALUE
                        && (dist[u]+graph[u][v])<dist[v]){
                    dist[v] = dist[u] + graph[u][v];
                }
            }
        }
        printSolution(dist,V);
    }
}
