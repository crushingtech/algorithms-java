package com.coursera.dp;

/**
 * Created by eshulga on 3/11/16.
 */
public class KnapSack {
    public static void main(String[] args) {
        int[] val = {5, 20, 3, 6};
//        System.out.println(new KnapSack().run(50, new int[]{10, 20, 30}, val, val.length));
        int[] weights = {2, 5, 2, 3};
        System.out.println(new KnapSack().runDP(4, weights, val, val.length));
        System.out.println(new KnapSack().runDPKnapsack(5, weights, val));
    }

    private int runDPKnapsack(int W, int[] weights, int[] val) {
        int[] dp = new int[W + 1];

        for (int i = 1; i <= W; i++) {
            for (int j = 0; j < weights.length; j++) {
                if (weights[j] <= i) {
                    dp[i] = Math.max(dp[i],Math.max(val[j], dp[i - weights[j]] + val[j]));
                }
            }
        }
        for (int i = 0; i < dp.length; i++) {
            System.out.print(dp[i] + " ");
        }
        System.out.println();
        for (int i = dp.length-1; i > 0; i--) {
            if(dp[i] - dp[i-1]>0) {
                System.out.print(dp[i] - dp[i - 1] + " ");
            }
        }
        System.out.println();
        return dp[W];
    }

    private int runDP(int W, int[] weights, int[] val, int n) {
        int[][] dp = new int[n + 1][W + 1];
        for (int i = 0; i <= n; i++) {
            for (int w = 0; w <= W; w++) {
                if (i == 0 || w == 0) {
                    dp[i][w] = 0;
                } else if (weights[i - 1] <= w) {
                    dp[i][w] = Math.max(
                            val[i - 1] + dp[i - 1][w - weights[i - 1]],
                            dp[i - 1][w]);
                } else {
                    dp[i][w] = dp[i - 1][w];
                }
            }
        }
        for (int i = 0; i <= n; i++) {
            for (int j = 0; j <= W; j++) {
                System.out.print(dp[i][j] + " ");
            }
            System.out.println();
        }
        return dp[n][W];
    }

    private int run(int W, int[] wt, int[] val, int n) {
        if (W == 0 || n == 0) {
            return 0;
        }
        if (wt[n - 1] > W) {
            return run(W, wt, val, n - 1);
        }
        return Math.max(
                run(W, wt, val, n - 1),
                val[n - 1] + run(W - wt[n - 1], wt, val, n - 1)
        );
    }

    public int runReq(int W, int[] weights, int[] val, int n) {
        if (W == 0 || n == 0) {
            return 0;
        }
        if (weights[n - 1] > W) {
            return runReq(W, weights, val, n - 1);
        }

        return Math.max(
                runReq(W, weights, val, n - 1),
                val[n - 1] + runReq(W - weights[n - 1], weights, val, n - 1)
        );
    }
}
