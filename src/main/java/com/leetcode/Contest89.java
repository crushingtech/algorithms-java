package com.leetcode;

public class Contest89 {

    public static void main(String[] args) {
        System.out.println(new Contest89().peakIndexInMountainArray(new int[]{0, 2, 1}));
        System.out.println(new Contest89().peakIndexInMountainArray(new int[]{0, 2, 1, 0}));
    }

    public int peakIndexInMountainArray(int[] A) {
        for (int i = 1; i < A.length - 1; i++) {
            if (A[i] > A[i - 1] && A[i] > A[i + 1]){
                return i;
            }
        }
        return -1;
    }
}
