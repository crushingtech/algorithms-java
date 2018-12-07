package com.book;

import java.util.*;

/**
 * Created by eshulga on 3/29/17.
 */
public class DPGreedy {
    //1, 2, 3, 4, 5  [1, 2, 3], [2, 3, 4] [3, 4, 5] and [1, 2, 3, 4], [2, 3, 4, 5] [1, 2, 3, 4, 5]
    public static void main(String[] args) {
//        System.out.println(new DPGreedy().findContentChildren(new int[]{10,9,8,7}, new int[]{5,6,7,8}));
        //TODO
//        System.out.println(new DPGreedy().findMaximizedCapital(1, 2, new int[]{1,2,3}, new int[]{0, 2, 3}));
//        System.out.println(new DPGreedy().findMaximizedCapital(2, 2, new int[]{1,2,3}, new int[]{0, 2, 3}));
        System.out.println(new DPGreedy().findTargetSumWays(new int[]{1, 1, 1, 1, 1}, 3));
    }

    public int findTargetSumWays(int[] nums, int S) {
        return 0;
    }

    public int findMaximizedCapital(int k, int W, int[] Profits, int[] Capital) {
        if(Profits.length==0 || k==0 || Capital.length==0) return 0;
        List<Integer> ar = new ArrayList<>();
        for (int i = 0; i < Profits.length; i++) {
            ar.add(i);
        }
        Collections.sort(ar, new Comparator<Integer>() {
            @Override
            public int compare(Integer integer, Integer t1) {
                return Double.compare(Profits[t1]/Double.valueOf(Capital[t1]),Profits[integer]/Double.valueOf(Capital[integer]));
            }
        });
        int i=0;
        int j=0;
        int totalProfit = 0;
        while(i<k && W>0 && j<ar.size()){
            int index = ar.get(j);
            if(W>=Capital[index]) {
                W -=Capital[index];
                totalProfit+=Profits[index];
                i++;
            }
            j++;
        }
        return totalProfit;
    }


    public int numberOfArithmeticSlices(int[] A) {
        if (A.length < 3) return 0;
        int res = 0;
        int sum = 0;
        for (int i = 2; i < A.length; i++) {
            if (Math.abs(A[i] - A[i - 1]) == Math.abs(A[i - 1] - A[i - 2])) {
                res++;
                sum += res;
            } else {
                res = 0;
            }
        }
        return sum;
    }

    public int findContentChildren(int[] g, int[] s) {
        if (s.length == 0 || g.length == 0) return 0;
        Arrays.sort(g);
        Arrays.sort(s);
        int res = 0;
        int j = 0;
        for (int i = 0; i < g.length; i++) {
            while (j < s.length && s[j] < g[i]) j++;
            if (j >= s.length) return res;
            if (g[i] <= s[j]) {
                res++;
                j++;
            }
        }
        return res;
    }
}
