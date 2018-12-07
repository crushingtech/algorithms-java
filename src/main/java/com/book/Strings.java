package com.book;

public class Strings {

    public static void main(String[] args) {
//        System.out.println(new Strings().customSortString("s", "sddsdcsba"));
        System.out.println(new Strings().scoreOfParentheses("(())"));
        System.out.println(new Strings().scoreOfParentheses("(()(()))"));
    }

    //https://leetcode.com/problems/daily-temperatures/description/
    public int[] dailyTemperatures(int[] temperatures) {
        return new int[]{};
    }


    //https://leetcode.com/problems/score-of-parentheses/description/
    public int scoreOfParentheses(String S) {
        return 0;
    }

    public String customSortString(String S, String T) {
        if (S.length() == 0) return T;
        int[] counts = new int[26];
        for (int i = 0; i < T.length(); i++) {
            counts[T.charAt(i) - 'a']++;
        }
        char[] res = new char[T.length()];
        int c = 0;
        for (int i = 0; i < S.length(); i++) {
            while (counts[S.charAt(i) - 'a']-- > 0) res[c++] = S.charAt(i);
        }
        for (int i = 0; i < counts.length; i++) {
            while (counts[i]-- > 0) res[c++] = (char) (i + 'a');
        }
        return new String(res);
    }
}
