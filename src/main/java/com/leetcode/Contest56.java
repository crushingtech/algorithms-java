package com.leetcode;

import java.util.*;

public class Contest56 {

    public static void main(String[] args) {
//        System.out.println(new Contest56().isOneBitCharacter(new int[]{1, 0, 0}));
//        System.out.println(new Contest56().isOneBitCharacter(new int[]{111, 0}));

//        System.out.println(new Contest56().compress(new char[]{'a','a','b','b','c','c','c'}));
//        System.out.println(new Contest56().compress(new char[]{'a'})==1);
//        System.out.println(new Contest56().compress(new char[]{'a','b'})==2);
//        System.out.println(new Contest56().compress(new char[]{'a','b','b','b','b','b','b','b','b','b','b','b','b'})==4);
//        System.out.println(new Contest56().compress(new char[]{'a','b','b','b','b','b','b','b','b','b','b','b','b','a','a','a','a'})==6)

        System.out.println(new Contest56().findLength(new int[]{1, 2, 3, 2, 1, 12, 13, 14, 15, 16, 20}, new int[]{3, 2, 1, 100, 12, 13, 14, 15, 4, 7}));
        System.out.println(new Contest56().findLength(new int[]{1, 4, 2}, new int[]{1}));
        System.out.println(new Contest56().findLength(new int[]{0, 0, 0, 0, 0}, new int[]{0, 0, 0, 0, 0}));

//        System.out.println(new Contest56().smallestDistancePair(new int[]{1, 3, 1, 4}, 2));
    }

    public int smallestDistancePair(int[] nums, int k) {
        PriorityQueue<Integer> pq = new PriorityQueue<>(Collections.reverseOrder());
        for (int i = 0; i < nums.length; i++) {
            for (int j = i + 1; j < nums.length; j++) {
                int distance = Math.abs(nums[i] - nums[j]);
                pq.add(distance);
                if (pq.size() > k) pq.poll();
            }
        }
        return pq.poll();
    }

    public int findLengthDP(int[] A, int[] B) {
        int[][] dp = new int[A.length][B.length];
        int maxGlobal = 0;
        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < B.length; j++) {
                if (A[i] == B[j]) {
                    dp[i][j] = 1;
                    if (i != 0 && j != 0) {
                        dp[i][j] += dp[i - 1][j - 1];
                    }
                }
                maxGlobal = Math.max(maxGlobal, dp[i][j]);
            }
        }
        return maxGlobal;
    }

    public int findLength(int[] A, int[] B) {
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int i = 0; i < B.length; i++) {
            map.computeIfAbsent(B[i], x->new ArrayList()).add(i);
        }
        int maxGlobal = 0;
        for (int i = 0; i < A.length; i++) {
            if (!map.containsKey(A[i])) continue;
            List<Integer> indexes = map.get(A[i]);
            for (Integer startB : indexes) {
                int offset = 1;
                while (startB + offset < B.length && i + offset < A.length && B[startB + offset] == A[i + offset]) {
                    offset++;
                }
                maxGlobal = Math.max(maxGlobal, offset);
            }
        }
        return maxGlobal;
    }


    public boolean isOneBitCharacter(int[] bits) {
        for (int i = 0; i < bits.length; i += 2) {

        }
        return false;
    }

    public int compress(char[] chars) {
        int cPosition = 0;
        char current = chars[0];
        int count = 1;
        for (int i = 1; i < chars.length; i++) {
            if (chars[i] != current) {
                char newCurrent = chars[i];
                if (count == 1) {
                    chars[cPosition++] = current;
                } else {
                    chars[cPosition++] = current;
                    String countS = count + "";
                    for (int j = 0; j < countS.length(); j++) {
                        chars[cPosition++] = countS.charAt(j);
                    }
                }
                current = newCurrent;
                count = 1;
            } else {
                count++;
            }
        }
        if (count == 1) {
            chars[cPosition++] = current;
        } else {
            chars[cPosition++] = current;
            String countS = count + "";
            for (int j = 0; j < countS.length(); j++) {
                chars[cPosition++] = countS.charAt(j);
            }
        }
        return cPosition;
    }


}
