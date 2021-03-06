package com.algorithms.permutations;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by eshulga on 10/11/16.
 * P(n,r)=n!(n−r)!
 * n^r
 */
public class DfsPermutations {
    public static void main(String[] args) {
        List<String> permute = new DfsPermutations().permute("1234", 4);
        System.out.println(permute);
        System.out.println("Total: " + permute.size());
    }

    private List<String> permute(String s, int k) {
        boolean[] marked = new boolean[s.length() + 1];
        List<String> res = new ArrayList<>();
        StringBuffer sb = new StringBuffer();
        dfs(s, k, sb, marked, res);
        return res;
    }

    private void dfs(String str, int k, StringBuffer sb, boolean[] marked, List<String> res) {
        if (sb.length() == k) {
            res.add(sb.toString());
            return;
        }
        for (int i = 0; i < str.length(); i++) {
            //if element is not visited yet
            if (!marked[i]) {
                //if input string has duplicates, ignore them and continue
                if (i > 0 && str.charAt(i) == str.charAt(i - 1)) {
                    continue;
                }
                sb.append(str.charAt(i));
                marked[i] = true;
                // recursively go to next element
                dfs(str, k, sb, marked, res);
                sb.deleteCharAt(sb.length() - 1);
                marked[i] = false;
            }
        }
    }
}
