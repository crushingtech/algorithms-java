package com.algorithms.arrays;

/**
 * Shift array of integers circularly, given input array and shift size.
 * O(N) time and space complexity.
 * <p>
 * Created by ievgen on 9/29/2014.
 */
public class CircularShiftArray {

    public static void main(String[] args) {
        int[] ar = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        int[] newAr = new CircularShiftArray().shiftToLeft(5, ar);
        for (int i = 0; i < newAr.length; i++) {
            System.out.println(newAr[i]);
        }
    }

    public int[] shiftToLeft(int shift, int[] ar) {
        int[] shifted = new int[ar.length];
        int rightShift = ar.length - (shift % ar.length);
        for (int i = 0; i < ar.length; i++) {
            int position = (i + rightShift) % ar.length;
            shifted[position] = ar[i];
        }
        return shifted;
    }

    public int[] shiftToRight(int shift, int[] ar) {
        int[] shifted = new int[ar.length];
        for (int i = 0; i < ar.length; i++) {
            int position = (shift + i) % ar.length;
            shifted[position] = ar[i];
        }
        return shifted;
    }
}
