import java.util.*;

public class LeetCode {


    public static void main(String[] args) {
        System.out.println(new LeetCode().buddyStrings("ab","ba")==true);
        System.out.println(new LeetCode().buddyStrings("ab","ab")==false);
        System.out.println(new LeetCode().buddyStrings("aa","aa")==true);
        System.out.println(new LeetCode().buddyStrings("aaaaaaabc","aaaaaaacb")==true);
        System.out.println(new LeetCode().buddyStrings("","aa")==false);

    }

    public boolean buddyStrings(String A, String B) {
        if(A==null || B==null) return false;
        if(A.length()!=B.length()) return false;
        Set<Character> set = new HashSet<>();
        for (int i = 0; i < A.length(); i++) {
            set.add(A.charAt(i));
        }
        if(A.equals(B) && set.size()<A.length()) return true;

        List<Integer> diff = new ArrayList<>();
        for (int i = 0; i < A.length(); i++) {
            if(A.charAt(i)!=B.charAt(i)) diff.add(i);
        }

        return diff.size()==2 && A.charAt(diff.get(0))==B.charAt(diff.get(1)) && A.charAt(diff.get(1))==B.charAt(diff.get(0));
    }

    private boolean myMethod() {
        return false;
    }


}
