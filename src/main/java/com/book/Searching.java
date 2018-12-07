package com.book;

import java.util.List;
import java.util.Random;

public class Searching {

    public static void main(String[] args) {
//        System.out.println(new com.book.Searching().findFirstOccurrence(new int[]{1,2,2,3,5,5,5},5));
//        System.out.println(new com.book.Searching().findFirstElGreaterThan(new int[]{1,2,2,3,5,5,5},4));
//        System.out.println(new com.book.Searching().findEntryEqualsIndex(new int[]{1,2,2,3,5,5,5}));
//        System.out.println(new com.book.Searching().findSmallestCyclicallySortedUniqueEl(new int[]{0, 4, 5, 7, 9}));
//        System.out.println(new com.book.Searching().findSmallestCyclicallySortedUniqueEl(new int[]{4, 5, 7, 9, 1, 2}));
//        System.out.println(new com.book.Searching().findSmallestCyclicallySortedDupEl(new int[]{4, 5, 5, 7, 9, 9, 1, 2}, 0, 7));
//        System.out.println(new com.book.Searching().findInCyclicallySorted(new int[]{4, 5, 5, 7, 9, 9, 1, 2},7));
//        System.out.println(new com.book.Searching().squareRoot(15));
//        System.out.println(new com.book.Searching().squareRootDouble(20));
//        System.out.println(new com.book.Searching().squareRootDouble(20));
    }


    private int findKLargest(int[] nums, int k) {
        int left = 0;
        int right = nums.length - 1;
        Random rand = new Random();
        while (left <= right) {
            int pivot = rand.nextInt(right - left + 1) + left;
            int newPivotIndex = partitionAround(nums, left, right, pivot);
            if (newPivotIndex == k - 1) {
                return nums[newPivotIndex];
            } else if (newPivotIndex > k - 1) {
                right = newPivotIndex - 1;
            } else {
                left = newPivotIndex + 1;
            }
        }
        return -1;
    }

    private int partitionAround(int[] nums, int left, int right, int pivot) {
        int newPivot = left;
        swap(nums, pivot, right);
        for (int i = left; i < right; i++) {
            if (nums[i] > nums[pivot]) {
                swap(nums, i, pivot++);
            }
        }
        swap(nums, right, newPivot);
        return newPivot;
    }

    private void swap(int[] ar, int i, int j) {
        int tmp = ar[i];
        ar[i] = ar[j];
        ar[j] = tmp;
    }

    private double squareRootDouble(int num) {
        double l, r;
        if (num < 1.0) {
            l = num;
            r = 1;
        } else {
            l = 0;
            r = num;
        }

        while (squareRootCompare(l, r) != 0) {
            double m = l + (r - l) / 2;
            if (squareRootCompare(m * m, num) == 0) {
                return m;
            } else if (squareRootCompare(m * m, num) < 0) {
                l = m;
            } else {
                r = m;
            }
        }
        return l;
    }

    private int squareRootCompare(double a, double b) {
        final double EPSILON = 0.00001;
        double diff = (a - b) / b;
        return diff < -EPSILON ? -1 : (diff > EPSILON ? 1 : 0);
    }

    private int squareRoot(int num) {
        int l = 0;
        int r = num;
        while (l <= r) {
            int m = l + (r - l) / 2;
            if (m * m < num) {
                l = m + 1;
            } else {
                r = m - 1;
            }
        }
        return l - 1;
    }

    private int findInCyclicallySorted(int[] nums, int target) {
        int l = 0;
        int r = nums.length - 1;
        while (l <= r) {
            int m = l + (r - l) / 2;
            if (target == nums[m]) {
                return m;
            } else if (target > nums[m] && target <= nums[r]) {
                l = m + 1;
            } else {
                r = m - 1;
            }
        }
        return -1;
    }

    private int findSmallestCyclicallySortedDupEl(int[] nums, int l, int r) {
        if (l == r) {
            return l;
        }
        int m = l + (r - l) / 2;
        if (nums[m] < nums[r]) {
            return findSmallestCyclicallySortedDupEl(nums, l, m);
        } else if (nums[m] > nums[r]) {
            return findSmallestCyclicallySortedDupEl(nums, m + 1, r);
        } else {
            int left = findSmallestCyclicallySortedDupEl(nums, l, m);
            int right = findSmallestCyclicallySortedDupEl(nums, m + 1, r);
            return Math.min(nums[left], nums[right]);
        }

    }

    private int findSmallestCyclicallySortedUniqueEl(int[] nums) {
        int s = 0;
        int right = nums.length - 1;
        while (s < right) {
            int m = s + (right - s) / 2;
            if (nums[m] < nums[right]) {
                right = m;
            } else {
                s = m + 1;
            }
        }
        return nums[s];
    }

    private int findEntryEqualsIndex(int[] nums) {
        int l = 0;
        int r = nums.length - 1;
        int res = -1;
        while (l <= r) {
            int m = l + (r - l) / 2;
            int diff = nums[m] - m;
            if (diff == 0) {
                return m;
            } else if (diff > 0) {
                r = m - 1;
            } else {
                l = m + 1;
            }
        }
        return res;
    }

    private int findFirstElGreaterThan(int[] nums, int target) {
        int l = 0;
        int r = nums.length - 1;
        int res = -1;
        while (l <= r) {
            int m = l + (r - l) / 2;
            if (nums[m] < target) {
                l = m + 1;
            } else {
                res = m;
                r = m - 1;
            }
        }
        return res;
    }

    private int findFirstOccurrence(int[] nums, int target) {
        int l = 0;
        int r = nums.length - 1;
        int res = -1;
        while (l <= r) {
            int m = l + (r - l) / 2;
            if (nums[m] == target) {
                res = m;
                r = m - 1;
            } else if (nums[m] < target) {
                l = m + 1;
            } else {
                r = m - 1;
            }
        }
        return res;
    }
}
