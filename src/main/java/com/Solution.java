package com;

import java.io.*;
import java.util.*;
import java.util.stream.IntStream;

// backtracking https://discuss.leetcode.com/topic/46161/a-general-approach-to-backtracking-questions-in-java-subsets-permutations-combination-sum-palindrome-partitioning/6

// (5,9) (4,5) (4,7)
public class Solution {

    public static void main(String[] args) {
//        InputStream inputStream = System.in;
//        OutputStream outputStream = System.out;
//        InputReader in = new InputReader(inputStream);
//        PrintWriter out = new PrintWriter(outputStream);
//        TaskA solver = new TaskA();
//        solver.solve(in, out);
//        out.close();
//        "[\"X..X\",\"...X\",\"...X\"]";
//        int[][] input = new int[6][2];
//        input[0] = new int[]{7, 0};
//        input[1] = new int[]{4, 4};
//        input[2] = new int[]{7, 1};
//        input[3] = new int[]{5, 0};
//        input[4] = new int[]{6, 1};
//        input[5] = new int[]{5, 2};
//        System.out.println(new Solution().reconstructQueue(input));
//        //[[46,89],[50,53],[52,68],[72,45],[77,81]]
//        int[][] envelopes = new int[5][2];
//        envelopes[0] = new int[]{46,89};
//        envelopes[1] = new int[]{50,53};
//        envelopes[2] = new int[]{52,68};
//        envelopes[3] = new int[]{72,45};
//        envelopes[4] = new int[]{77,81};
//        System.out.println(new Solution().maxEnvelopes(envelopes));
//        System.out.println(new Solution().maxProduct(new int[]{2, 3,-2,4}));
//        System.out.println(new Solution().maxProduct(new int[]{2, 3,-2,2,-2}));
//        //[1,2,3,4], return [24,12,8,6]
//        int[] ints = new Solution().productExceptSelf(new int[]{1, 2, 3, 4});
//        for (int i = 0; i < ints.length; i++) {
//            System.out.print(ints[i]+" ");
//        }
        //[1,2,3]  =>  [2,3,3]  =>  [3,4,3]  =>  [4,4,4] 0, 0, 1, 6, 8
//        System.out.println(new Solution().minMoves(new int[]{1, 0, 0, 8, 6}));
//        int[] ar = {1, 2, 0, 8, 6};
//        System.out.println(new Solution().findKth(ar,0,ar.length-1,3));
//        System.out.println(new Solution().topKFrequent(new int[]{1, 1, 1, 2, 2, 3}, 2));

//        System.out.println(new Solution().maxProfit(new int[]{7, 1, 5, 3, 6, 4,0,4}));
//        System.out.println(new Solution().find132pattern(new int[]{-2,1,2,-2,1,2}));
//        System.out.println(new Solution().isValidSerialization("9,3,4,#,#,1,#,#,2,#,6,#,#"));
//        System.out.println(new Solution().isValidSerialization("1,#"));
//        System.out.println(new Solution().isValidSerialization("9,#,#,1"));
//        System.out.println(new Solution().isValidSerialization("9,8,#,#"));
//        System.out.println(new Solution().isValid("{he}[{sa{d}]"));
//        System.out.println(new Solution().isPalindrome("race a car"));
//        System.out.println(new Solution().isPalindrome("A man, a plan, a canal: Panama"));
//        System.out.println(new Solution().hammingDistance(4, 14));
//        System.out.println(new Solution().findTheDifference("hello", "qolehl"));
//        int[] intersect = new Solution().intersect(new int[]{1}, new int[]{1,1});
//        System.out.println(new Solution().singleNumber(new int[]{1, 2, 3, 4, 3, 1, 2, 5, 5}));
//        System.out.println(new Solution().isSameTree(getTree(), getTree()));
//        TreeNode treeNode = new Solution().sortedArrayToBST(new int[]{1, 2, 3, 4, 5, 6, 7 });
//        System.out.println(new Solution().isSameTree(getTree(), getTree()));
//        TreeNode treeNode = new Solution().sortedListToBSTIterative(getSortedList());
//        System.out.println(new Solution().isValidBSTIteratively(getTree()));
//        System.out.println(new Solution().rob(new int[]{1, 3, 9, 7, 3, 8}));
//        System.out.println(new Solution().climbStairs(5));
//        System.out.println(new Solution().climbStairs2(5));
//        System.out.println(new Solution().integerBreak(11));
//        System.out.println(new Solution().lengthOfLIS(new int[]{1,3,6,7,9,4,10,5,6}));
//        System.out.println(new Solution().isSubsequence("acb", "ahbgdc"));
//        System.out.println(new Solution().coinChange(new int[]{2}, 3));
//        System.out.println(new KnapSack().run(50, new int[]{10, 20, 30}, val, val.length));
//        System.out.println(new Solution().findDuplicate(new int[]{1,2,3,4,6,8,9,0,1,1,100}));
//        System.out.println(new Solution().canPartition(new int[]{3,3,2, 3, 3,1,1}));
//        System.out.println(new Solution().numberOfwaysToFormNumber(new int[]{8, 3, 3,1,1},8));
//        System.out.println(new Solution().countNumberOfScores(7, new int[]{7, 3, 2}));
//        System.out.println(new Solution().letterCombinations("23"));
//        System.out.println(new Solution().combinationSumNum(new int[]{2, 3, 6, 7}, 7));
//        System.out.println(new Solution().search(new int[]{3},3));
//        System.out.println(new Solution().canCompleteCircuit(new int[]{3, 6, 4}, new int[]{2, 8, 3}));
//        System.out.println(new Solution().eraseOverlapIntervals(new Interval[]{new Interval(1,100),new Interval(11,22),new Interval(1,11),new Interval(2,12)}));
//        System.out.println(new Solution().findMinArrowShots(new int[][]{
//                new int[]{1,8},
//                new int[]{9,10}
//        }));
        TreeNode node = new TreeNode(10);
        node.left = new TreeNode(5);
        node.right = new TreeNode(50);
        node.right.right = new TreeNode(500);
        node.right.right.right = new TreeNode(5000);
        node.left.left = new TreeNode(3);
        node.left.right = new TreeNode(2);
        node.left.left.left = new TreeNode(3);
        node.left.left.right = new TreeNode(-2);
        node.left.left.right.left = new TreeNode(-10);
//        System.out.println(new Solution().pathSum(node,15));
//        node.right = new TreeNode(5);
//        node.right.right = new TreeNode(2);
//        System.out.println(new Solution().largestValues(node));
//        System.out.println(new Solution().wiggleMaxLength(new int[]{1, 17, 5,10,13,15,10,5,16,8}));
//        System.out.println(new Solution().findAnagrams("cbaebabacd","abc" ));
//        System.out.println(new Solution().minWindow("ADOBECODEBANC","ABC"));
//        System.out.println(new Solution().minWindow("A","AA"));
//        System.out.println(new Solution().longestPalindromeSubseq("cbbd"));
//        System.out.println(new Solution().longestPalindromeSubseqMemoriz("cbbd"));
//        System.out.println(new Solution().longestCommonSubstring("sweden", "aswem"));
//        System.out.println(new Solution().longestSubstringWithoutRepeatedCharacters("abccdjksp"));
//        System.out.println(new Solution().readBinaryWatch(2));
//        System.out.println(new Solution().findKthLargest(new int[]{4, 1, 2,7,3,93,5}, 3));
//        System.out.println(new Solution().findMedian(new int[]{4, 1, 2,7,3}));
        //Input: [ [1,4], [2,3], [3,4] ] 1-5 2-3 5-6 [[1,12],[2,9],[3,10],[13,14],[15,16],[16,17]]
//        System.out.println(new Solution().findRightInterval(new Interval[]{new Interval(1,12),new Interval(2,9),new Interval(3,10),new Interval(13,14),new Interval(15,16),new Interval(16,17)}));
//        System.out.println(new Solution().removeDuplicates(new int[]{1,4,3}));
//        int[][] x = new Solution().matrixReshape(new int[][]{
//                new int[]{1, 2, 3},
//                new int[]{4, 5, 6},
//                new int[]{7, 8, 9},
//                new int[]{10, 11, 12}
//        }, 2, 6);
//        printMatrix(x);
//          System.out.println(new Solution().subsetsWithDup(new int[]{1,2,2}));
//          System.out.println(new Solution().findDuplicates(new int[]{4,3,2,7,8,2,3,1}));
//          System.out.println(new Solution().pattern132(new int[]{1, 2, 3, 4}));
//        System.out.println(new Solution().distributeCandies(new int[]{1,1,2,3}));
        for (int i = 0; i < 17; i++) {
//            System.out.println(i+":===:"+new Solution().isPerfectSquare(i));
//            System.out.println(i+":===:"+new Solution().judgeSquareSum(i));
        }
        //Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
        //Output: 7 -> 0 -> 8
        ListNode node1 = new ListNode(2);
        node1.next = new ListNode(4);
        node1.next.next = new ListNode(3);
        node1.next.next.next = new ListNode(4);
        node1.next.next.next.next = new ListNode(2);

        ListNode listNode2 = new ListNode(5);
        listNode2.next = new ListNode(6);
        listNode2.next.next = new ListNode(4);
        int[] nums = {1,0};
//        System.out.println(new Solution().threeSum(new int[]{1,2}));
        for (int i = 0; i < nums.length; i++) {
//            System.out.println(nums[i]);
        }

//        System.out.println(new Solution().countBattleships(new char[][]{
//                new char[]{'X','.','.','X'},
//                new char[]{'.','.','.','X'},
//                new char[]{'.','.','.','X'}
//        }));
//        System.out.println(new Solution().numSubarrayProductLessThanK(new int[]{10, 5, 2, 6},3));
//        System.out.println(new Solution().canPartitionKSubsets(new int[]{3, 1, 1, 2, 2, 1, 2},4));
//        System.out.println(new Solution().canPartition(new int[]{1,5,11,5}));
//        List<List<Integer>> lists = new Solution().fourSum(new int[]{-3,-2,-1,0,0,1,2,3}, 0);
//        for (List<Integer> list : lists) {
//            for (Integer integer : list) {
//                System.out.print(integer);
//            }
//            System.out.println();
//        }
//        System.out.println(new Solution().findKthLargest(new int[]{2,1,3,0,5},3));
//        System.out.println(new  Solution().topKFrequent(new int[]{1,1,1,2,2,3},2));
//        System.out.println(new Solution().findCircleNum(new int[][]{
//                new int[]{1,1,0},
//                new int[]{1,1,0},
//                new int[]{0,0,0}
//        }));
        TreeNode n = new TreeNode(5);
        n.left = new TreeNode(2);
        n.right= new TreeNode(13);
//        n.right.right= new TreeNode(3);
//        n.right.left= new TreeNode(3);
//        n.left.left.left = new TreeNode(3);
//        n.right = new TreeNode(1);
//        System.out.println(new  Solution().kthSmallest(n, 3));
//        System.out.println(new  Solution().kthSmallest2(n, 3));
//        System.out.println(new  Solution().longestWord(new String[]{"aanana","a", "banana", "app", "appl", "ap", "apply", "apple"}));

//        System.out.println(new  Solution().compress(new char[]{'a','b','b','b','b','b','b','b','b','b','b','b','b'}));
//        System.out.println(new  Solution().smallestDistancePair(new int[]{1,6,1},3));
//        List<int[]> x = new Solution().kSmallestPairs(new int[]{1, 7, 11}, new int[]{2, 4, 6}, 9);
//        x.forEach(r->{
//            System.out.println(r[0]+" "+r[1]);
//        });
        // ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].
//        System.out.println(new  Solution().letterCombinations2("23"));
        //[7], [2, 2, 3]
//        System.out.println(new  Solution().combinationSum(new int[]{2, 3, 6, 7},7));
        //  [1, 7],[1, 2, 5],[2, 6],[1, 1, 6]
//        System.out.println(new  Solution().combinationSum2(new int[]{10, 1, 2, 7, 6, 1, 5},8));
        //[3], [1], [2], [1,2,3], [1,3], [2,3], [1,2], []
//        System.out.println(new  Solution().subsets(new int[]{1,2,3}));
//        System.out.println(new  Solution().isSymmetric(n));
//        printMatrix(new Solution().updateMatrix(new int[][]{
//                new int[]{0,1,1},
//                new int[]{1,1,1},
//                new int[]{1,1,1}
//        }));

//        System.out.println(new Solution().numIslands(new char[][]{
//                new char[]{'1','1','1','1','0'},
//                new char[]{'1','1','0','1','0'},
//                new char[]{'1','1','0','0','0'},
//                new char[]{'0','0','0','0','0'}
//        }));[['X','O','X','X'],['O','X','O','X'],['X','O','X','O'],['O','X','O','X'],['X','O','X','O'],['O','X','O','X']]
//        new Solution().solve(new char[][]{
//                new char[]{'X','O','X','X'},
//                new char[]{'O','X','O','X'},
//                new char[]{'O','X','O','X'},
//                new char[]{'X','O','X','O'},
//                new char[]{'O','X','O','X'}
//        });
//        System.out.println(new Solution().maxAreaOfIsland(new int[][]{
//                new int[]{1,0,0},
//                new int[]{1,0,0},
//                new int[]{1,1,1}
//        }));

//        new Solution().convertBST(n);
        //"((()))", "(()())", "(())()", "()(())", "()()()"
//        System.out.println(new Solution().generateParenthesis(3));
        //DP and memorization can be used
//        System.out.println(new Solution().findTargetSumWays(new int[]{1, 1, 1, 1, 1},3));
//        System.out.println(new Solution().pivotIndex(new int[]{-1,-1,-1,-1,-1,0}));
//        System.out.println(new Solution().pivotIndex(new int[]{1, 7, 3, 6, 5, 6}));
//        System.out.println(new Solution().pivotIndex(new int[]{1,2,3}));
//        System.out.println(new Solution().pivotIndex(new int[]{-1,-1,-1,-1,-1,-1}));
//        System.out.println(new Solution().findDisappearedNumbers(new int[]{4,3,2,7,8,2,3,1}));
//        System.out.println(new Solution().findDuplicates(new int[]{4,3,2,7,8,2,3,1}));
//        System.out.println(new Solution().findDuplicates(new int[]{10,2,5,10,9,1,1,4,3,7}));
        System.out.println(new Solution().partitionLabels("ababcbacadefegdehijhklij"));
    }

    //Input: S = "ababcbacadefegdehijhklij"
    //Output: [9,7,8]
    //The partition is "ababcbaca", "defegde", "hijhklij".
    public List<Integer> partitionLabels(String S) {
        int[] ar = new int[256];
        for (int i = 0; i < S.length(); i++) {
            ar[S.charAt(i)] = Math.max(i,ar[S.charAt(i)]);
        }
        int lastMaxPos = 0;
        int max = 0;
        List<Integer> res = new ArrayList<>();
        for (int i = 0; i < S.length(); i++) {
            if(i==max) {
                res.add(max-lastMaxPos);
                lastMaxPos = max;
            }
            max = Math.max(ar[S.charAt(i)],max);
        }
        return res;
    }


    //Input:[4,3,2,7,8,2,3,1]
    //Output:[2,3]
    public List<Integer> findDuplicates(int[] nums) {
        List<Integer> res = new ArrayList();
        for (int i = 0; i < nums.length; i++) {
            int index = Math.abs(nums[i])-1;
            if(nums[index]<0) res.add(Math.abs(index+1));
            else nums[index]*=-1;
        }
        return res;
    }

    //Input:[4,3,2,7,8,2,3,1]
    //Output:[5,6]
    public List<Integer> findDisappearedNumbers(int[] nums) {
        for (int i = 0; i < nums.length; i++) {
            int index = Math.abs(nums[i])-1;
            if(nums[index]>0) nums[index]*=-1;
        }
        ArrayList<Integer> res = new ArrayList<>();
        for (int i = 0; i < nums.length; i++) {
            if(nums[i]>0) res.add(i+1);
        }
        return res;
    }

    public List<Integer> findDisappearedNumbers2(int[] nums) {
        List<Integer> ret = new ArrayList<Integer>();

        for(int i = 0; i < nums.length; i++) {
            int val = Math.abs(nums[i]) - 1;
            if(nums[val] > 0) {
                nums[val] = -nums[val];
            }
        }

        for(int i = 0; i < nums.length; i++) {
            if(nums[i] > 0) {
                ret.add(i+1);
            }
        }
        return ret;
    }


    //Given an array of integers, every element appears twice except for one. Find that single one.
    public int singleNumber(int[] nums) {
        return 2*IntStream.of(nums).distinct().sum() - IntStream.of(nums).sum();
    }


    public int pivotIndex(int[] nums) {
        if(nums.length==0) return -1;
        int l = 1;
        int r = nums.length-2;
        int lSum = nums[0];
        int rSum = nums[nums.length-1];
        int candidate = -1;
        while(l<=r){
            if(lSum==rSum && l==r){
                candidate=l;
            }
            if(nums[l]==0){
                l++;
                continue;
            }
            if(nums[r]==0){
                r--;
                continue;
            }
            if(Math.abs(lSum+nums[l]) < Math.abs(rSum + nums[r])) {
                lSum += nums[l++];
            }else {
                rSum += nums[r--];
            }

        }
        return candidate;
    }

    public int findTargetSumWays(int[] nums, int S) {
        int[] res = {0};
        dfsTargetSumWays(nums,S,0,res,0);
        return res[0];
    }

    private void dfsTargetSumWays(int[] nums, int S,int tmp, int[] res, int start) {
        if(start==nums.length) {
            if (tmp == S) {
                res[0]++;
            }
            return;
        }
        dfsTargetSumWays(nums,S,tmp-nums[start],res,start+1);
        dfsTargetSumWays(nums,S,tmp+nums[start],res,start+1);
    }


    public List<String> generateParenthesis(int n) {
        List<String> res = new ArrayList<>();
        dfsParanthesis(n,new StringBuffer(), res,0,0,0);
        return res;
    }

    private void dfsParanthesis(int max, StringBuffer sb,List<String> res, int maxL,int maxR, int start) {
        if(sb.length()==max*2){
            res.add(sb.toString());
            return;
        }
             if(maxL<max){
                 sb.append("(");
                 dfsParanthesis(max,sb,res,maxL+1,maxR,start+1);
                 sb.deleteCharAt(sb.length()-1);
             }
            if(maxR<maxL){
                sb.append(")");
                dfsParanthesis(max,sb,res,maxL,maxR+1,start+1);
                sb.deleteCharAt(sb.length()-1);
            }
    }

    public TreeNode convertBST(TreeNode root) {
        if(root==null || root.left==null && root.right==null) return root;
        inorderAlter(root,new int[]{0});
        return root;
    }

    private void inorderAlter(TreeNode root,int[] sum) {
        if(root==null) return;
        inorderAlter(root.right,sum);
        int tempSum = sum[0];
        sum[0]+=root.val;
        root.val+=tempSum;
        inorderAlter(root.left,sum);
    }

    private static int convert(TreeNode node,  int sum){
        if(node==null) return sum;
        sum = convert(node.right, sum);
        sum+=node.val;
        node.val = sum;
        return convert(node.left, sum);
    }




    public int maxAreaOfIsland(int[][] grid) {
        if(grid.length==0) return 0;
        boolean[][] visited = new boolean[grid.length][grid[0].length];
        int res = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if(!visited[i][j] && grid[i][j]==1) {
                    int[] count = {0};
                    dfsMaxArea(grid, visited, i, j, count);
                    res = Math.max(res, count[0]);
                }
            }
        }
        return res;
    }

    private void dfsMaxArea(int[][] grid, boolean[][] visited, int i, int j, int[] count) {
        visited[i][j] = true;
        count[0]++;
        List<int[]> neighbours = Arrays.asList(new int[]{i+1,j},new int[]{i-1,j},new int[]{i,j+1},new int[]{i,j-1});
        for (int[] neighbour : neighbours) {
            int newI = neighbour[0];
            int newJ = neighbour[1];
            if(newI>=0 && newJ>=0 && newI<grid.length && newJ < grid[0].length && grid[newI][newJ]==1 && !visited[newI][newJ]){
                dfsMaxArea(grid,visited,newI,newJ,count);
            }
        }
    }

    public void solve(char[][] board) {
        if(board.length==0 || (board.length==1 && board[0].length<=1)) return;
        boolean[][] visited = new boolean[board.length][board[0].length];
        int n = board.length;
        int m = board[0].length;
        for (int i = 0; i < m; i++) {
            if(board[0][i]=='O'){
                dfsEnclosing(board,visited,0,i);
            }
            if(board[n-1][i]=='O'){
                dfsEnclosing(board,visited,n-1,i);
            }
        }
        for (int i = 0; i < n; i++) {
            if(board[i][0]=='O'){
                dfsEnclosing(board,visited,i,0);
            }
            if(board[i][m-1]=='O'){
                dfsEnclosing(board,visited,i,m-1);
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if(!visited[i][j] && board[i][j]=='O'){
                    board[i][j] = 'X';
                }
            }
        }
    }

    private void dfsEnclosing(char[][] board, boolean[][] visited, int i, int j) {
        visited[i][j] = true;
        List<int[]> neighbours = Arrays.asList(new int[]{i+1,j},new int[]{i-1,j},new int[]{i,j+1},new int[]{i,j-1});
        for (int[] neighbour : neighbours) {
            int newI = neighbour[0];
            int newJ = neighbour[1];
            if(newI>=0 && newJ>=0 && newI<board.length && newJ<board[0].length && !visited[newI][newJ] && board[newI][newJ]=='O'){
                dfsEnclosing(board,visited,newI,newJ);
            }
        }
    }


    public int numIslands(char[][] grid) {
        if(grid.length==0) return 0;
        boolean[][] visited = new boolean[grid.length][grid[0].length];
        int c = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if(!visited[i][j] && grid[i][j]=='1'){
                    dfsNumIslands(grid,visited,i,j);
                    c++;
                }
            }
        }
        return c;
    }

    private void dfsNumIslands(char[][] m, boolean[][] visited, int i, int j) {
        visited[i][j] = true;
        List<int[]> neighbours = Arrays.asList(new int[]{i+1,j},new int[]{i-1,j},new int[]{i,j+1},new int[]{i,j-1});
        for (int[] n : neighbours) {
            int tnpI = n[0];
            int tmpJ = n[1];
            if(tnpI <m.length && tnpI >=0 && tmpJ <m[0].length && tmpJ >=0 && m[tnpI][tmpJ]=='1' && !visited[tnpI][tmpJ]){
                dfsNumIslands(m,visited, tnpI, tmpJ);
            }
        }
    }

    public int[][] updateMatrix(int[][] matrix) {
        int[][] res = new int[matrix.length][matrix[0].length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                if(matrix[i][j]!=0){
                    int steps = bfsUpdateMatrix(matrix, new int[]{i, j});
                    res[i][j] = steps;
                }
            }
        }
        return res;
    }

    private int bfsUpdateMatrix(int[][] matrix, int[] c) {
        Queue<int[]> q = new LinkedList<>();
        q.add(new int[]{c[0],c[1],0});
        while(!q.isEmpty()){
            int[] poll = q.poll();
            int steps = poll[2];
            if(matrix[poll[0]][poll[1]]==0) return steps;
            int x = poll[0];
            int y = poll[1];

            List<int[]> around = Arrays.asList(new int[]{x+1, y,steps+1},new int[]{x-1, y,steps+1},new int[]{x, y+1,steps+1},new int[]{x, y-1,steps+1});
            for (int[] n : around) {
                if(isValid(matrix,n[0],n[1])){
                    q.add(n);
                }
            }
        }
        return 0;
    }

    private boolean isValid(int[][] m, int i, int j) {
        return i<m.length && i>=0 && j<m[0].length && j>=0;
    }

    public boolean isSymmetric(TreeNode root) {
        if(root==null) return true;
        return isSymetricReq(root.left,root.right);
    }

    private boolean isSymetricReq(TreeNode left, TreeNode right) {
        if(left==null && right==null) return true;
        if(left==null || right==null) return false;
        return left.val== right.val
                && isSymetricReq(left.left,right.right) && isSymetricReq(left.right,right.left);
    }


    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        if(nums.length==0) return res;
        Arrays.sort(nums);
        subsetsReq(nums,new LinkedList<Integer>(),res,0);
        return res;
    }
    private void subsetsReq(int[] nums, LinkedList<Integer> temp, List<List<Integer>> res, int start) {
        res.add(new LinkedList<>(temp));
        for (int j = start; j < nums.length; j++) {
            temp.add(nums[j]);
            subsetsReq(nums,temp,res,j+1);
            temp.remove(temp.size()-1);
        }
    }

    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        Arrays.sort(candidates);
        List<List<Integer>> res = new ArrayList<>();
        if(candidates.length==0) return res;
        combinationSumReq2(candidates,target,new LinkedList<>(),0,res);
        return res;
    }
    private void combinationSumReq2(int[] candidates, int target, LinkedList<Integer> objects, int start, List<List<Integer>> res) {
        if(target<0) return;
        else if(target==0){
            res.add(new ArrayList<>(objects));
        }else{
            for (int j = start; j < candidates.length; j++) {
                if(j>0 && candidates[j]==candidates[j-1] && j!=start) continue;
                objects.add(candidates[j]);
                //j is passed along, not a start variable
                combinationSumReq2(candidates,target - candidates[j],objects, j+1, res);
                objects.remove(objects.size()-1);
            }
        }
    }

    public List<List<Integer>> combinationSum(int[] nums, int target) {
        List<List<Integer>> list = new ArrayList<>();
        Arrays.sort(nums);
        backtrack(list, new ArrayList<>(), nums, target, 0);
        return list;
    }

    private void backtrack(List<List<Integer>> list, List<Integer> tempList, int [] nums, int remain, int start){
        if(remain < 0) return;
        else if(remain == 0) list.add(new ArrayList<>(tempList));
        else{
            for(int i = start; i < nums.length; i++){
                tempList.add(nums[i]);
                backtrack(list, tempList, nums, remain - nums[i], i); // not i + 1 because we can reuse same elements
                tempList.remove(tempList.size() - 1);
            }
        }
    }


    public List<String> letterCombinations2(String digits) {
        List<String> res = new LinkedList<>();
        if(digits.isEmpty()) return res;
        String[] mapping = new String[] {"0", "1", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        letterCombinations2Req(0,digits,res,new StringBuffer(),mapping);
        return res;
    }

    private void letterCombinations2Req(int i, String digits, List<String> res, StringBuffer sb,String[] mapping) {
        if(sb.length()==digits.length()){
            res.add(sb.toString());
            return;
        }

        String button = mapping[Character.getNumericValue(digits.charAt(i))];
        for (int j = 0; j < button.length(); j++) {
            sb.append(button.charAt(j));
            letterCombinations2Req(i+1,digits,res,sb,mapping);
            sb.deleteCharAt(sb.length()-1);
        }
    }

    class TPair{
        int depth;
        TreeNode node;

        public TPair(int depth, TreeNode node) {
            this.depth = depth;
            this.node = node;
        }
    }

    public List<int[]> kSmallestPairs(int[] nums1, int[] nums2, int k) {
        int s1 = 0;
        int s2 = 0;
        List<int[]> res = new ArrayList<>();
        int s1Reset = 0;
        int s2Reset = 0;
        while((s1<nums1.length || s2 < nums2.length) && k>0){

            if(nums1[s1]<nums2[s2]){
                res.add(new int[]{nums1[s1],nums2[s2++]});
            }else{
                res.add(new int[]{nums2[s2],nums1[s1++]});
            }
            k--;
            if(s1>=nums1.length && s2<nums2.length){
                s1 = s1Reset;
                s2++;
                s1Reset++;
                s2Reset++;
            }

            if(s2>=nums2.length && s1<nums1.length){
                s2 = s2Reset;
                s1++;
                s2Reset++;
                s1Reset++;
            }
        }
        return res;
    }

    class QPair{
        int root;
        int next;

        public QPair(int root, int next) {
            this.root = root;
            this.next = next;
        }
    }
    public int smallestDistancePair(int[] nums, int k) {
        Arrays.sort(nums);
        PriorityQueue<QPair> pq = new PriorityQueue(Comparator.<QPair> comparingInt(r->(nums[r.next]-nums[r.root])));

        for (int i = 0; i+1 < nums.length; i++) {
            pq.add(new QPair(i,i+1));
        }
        while(--k>0){
            QPair poll = pq.poll();
            if(poll.next+1<nums.length){
                pq.add(new QPair(poll.root,poll.next+1));
            }
        }
        return nums[pq.peek().next] - nums[pq.peek().root];
    }

    private void createPairsDfs(int i,int[] nums, boolean[] visited, LinkedList<Integer> temp, PriorityQueue<List<Integer>> q,Set<LinkedList<Integer>> set) {
        if(temp.size()==2 && !set.contains(temp)){
            LinkedList<Integer> e = new LinkedList<>(temp);
            q.add(e);
            set.add(e);
            return;
        }
        for (int j = i; j < nums.length; j++) {
            if(!visited[j]) {
                int num = nums[j];
                visited[j] = true;
                temp.add(num);
                createPairsDfs(i+1,nums, visited, temp, q, set);
                visited[j] = false;
                temp.removeLast();
            }
        }
    }

    public int compress(char[] chars) {
        int c = 1;
        int index = 0;
        int i;
        for (i = 1; i < chars.length; i++) {
            if(chars[i]==chars[i-1]){
                c++;
            }else{
                chars[index++] = chars[i-1];
                if(c>1) {
                    Stack<Integer> q = new Stack<>();
                    while (c > 0) {
                        q.push(c % 10);
                        c /= 10;
                    }
                    while (!q.isEmpty()) {
                        chars[index++] = Character.forDigit(q.pop().intValue(),10);
                    }
                }
                c = 1;
            }
        }
        if(c==0) return index;
        chars[index++] = chars[i - 1];
        if(c>1) {
            Stack<Integer> q = new Stack<>();
            while (c > 0) {
                q.push(c % 10);
                c /= 10;
            }
            while (!q.isEmpty()) {
                chars[index++] = Character.forDigit(q.pop().intValue(),10);
            }
        }
        return index;
    }



    public List<List<String>> accountsMerge(List<List<String>> accounts) {
        Map<String,List<Integer>> emailToAccount = new HashMap<>();
        for (int i = 0; i < accounts.size(); i++) {
            List<String> account= accounts.get(i);
            for (int j = 1; j < account.size(); j++) {
                String email = account.get(j);
                if(!emailToAccount.containsKey(email)){
                    emailToAccount.put(email,new LinkedList<>());
                }
                emailToAccount.get(email).add(i);
            }
        }
        List<List<String>> res = new ArrayList<>();
        boolean[] visited = new boolean[accounts.size()];
        for (int i = 0; i < accounts.size(); i++) {
            Set<String> set = new TreeSet<>();
            dfsAccounts(i,accounts,emailToAccount,set,visited);
            if(!set.isEmpty()){
                ArrayList<String> acc = new ArrayList<>(set);
                acc.add(0,accounts.get(i).get(0));
                res.add(acc);
            }
        }
        return res;
    }

    private void dfsAccounts(Integer i, List<List<String>> accounts, Map<String, List<Integer>> emailToAccount, Set<String> set, boolean[] visited) {
        if(visited[i]) return;
        visited[i] = true;
        List<String> acc = accounts.get(i);
        for (int j = 1; j < acc.size(); j++) {
            String email = acc.get(j);
            set.add(email);
            List<Integer> accs = emailToAccount.get(email);
            for (Integer ac : accs) {
                dfsAccounts(ac,accounts,emailToAccount,set,visited);
            }
        }
    }

    public int kthSmallest(TreeNode root, int k) {
        int c = count(root.left);
        if(k<=c) {
            return kthSmallest(root.left,k);
        } else if(k>c+1) {
            return kthSmallest(root.right,k-c-1);
        }
        return root.val;
    }

    public int count(TreeNode node){
        if(node==null) return 0;
        return 1 + count(node.left) + count(node.right);
    }





    public void dfs(int[][] M, int[] visited, int i) {
        for (int j = 0; j < M.length; j++) {
            if (M[i][j] == 1 && visited[j] == 0) {
                visited[j] = 1;
                dfs(M, visited, j);
            }
        }
    }
    public int findCircleNum(int[][] M) {
        int[] visited = new int[M.length];
        int count = 0;
        for (int i = 0; i < M.length; i++) {
            if (visited[i] == 0) {
                dfs(M, visited, i);
                count++;
            }
        }
        return count;
    }

//    public int findCircleNum(int[][] M) {
//        boolean[][] visited = new boolean[M.length][M[0].length];
//        int res = 0;
//        for (int i = 0; i < M.length; i++) {
//            for (int j = 0; j < M[0].length; j++) {
//                if(!visited[i][j] && M[i][j]==1) {
//                    dfs(M, visited,new Coordinate(i,j));
//                    res++;
//                }
//            }
//        }
//        return res;
//    }
//
//    private void dfs(int[][] m, boolean[][] visited, Coordinate c) {
//        List<Coordinate> shift = Arrays.asList(new Coordinate(-1,0),new Coordinate(1,0),
//                new Coordinate(0,-1),new Coordinate(0,1));
//        for (Coordinate cc : shift) {
//            Coordinate next = new Coordinate(cc.x+c.x, cc.y+c.y);
//            if(isValidNodeGraph(next,m) && !visited[next.x][next.y]){
//                visited[next.x][next.y] = true;
//                dfs(m,visited, next);
//            }
//        }
//    }
//
//    private boolean isValidNodeGraph(Coordinate n, int[][] m) {
//        return n.x<m.length && n.x>=0 && n.y<m[0].length && n.y>=0 && m[n.x][n.y]==1;
//    }

    public List<Integer> topKFrequent(int[] nums, int k) {
        Map<Integer,Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            map.put(nums[i],map.getOrDefault(nums[i],0)+1);
        }
        List<Integer>[] counts = new LinkedList[nums.length + 1];
        map.forEach((key,v) -> {
            if(counts[v]==null) counts[v] = new LinkedList<>();
            counts[v].add(key);
        });
        List<Integer> res = new ArrayList<>();
        for (int i = counts.length - 1; i >= 0 && k>0; i--) {
            if(counts[i]==null) continue;
            if(counts[i].size()<=k) res.addAll(counts[i]);
            else res.addAll(counts[i].subList(0,k));
            k-=counts[i].size();
        }
        return res;
    }





    public List<Integer> topKFrequent2(int[] nums, int k) {
        Map<Integer, Integer> map = new HashMap();
        for (int i = 0; i < nums.length; i++) {
            map.put(nums[i], map.getOrDefault(nums[i], 0) + 1);
        }
        List[] buckets = new LinkedList[nums.length + 1];
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            if (buckets[entry.getValue()] == null) {
                buckets[entry.getValue()] = new LinkedList<>();
            }
            buckets[entry.getValue()].add(entry.getKey());
        }
        List<Integer> res = new ArrayList<>();
        for (int i = buckets.length - 1; i >= 0; i--) {
            if (buckets[i] == null) continue;
            if (buckets[i].size() <= k) res.addAll(buckets[i]);
            else res.addAll(buckets[i].subList(0, k));
            k -= buckets[i].size();
        }
        return res;
    }

    public int findKthLargest(int[] nums, int k) {
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        for (int i = 0; i < nums.length; i++) {
            pq.offer(nums[i]);
            if(pq.size()>k){
                pq.poll();
            }
        }
        return pq.peek();
    }

    public int findKthLargest2(int[] nums, int k) {
        int start = 0, end = nums.length - 1, index = k;
        while (start < end) {
            int pivot = partion(nums, start, end);
            if (pivot < index) start = pivot + 1;
            else if (pivot > index) end = pivot - 1;
            else return nums[pivot];
        }
        return nums[start];
    }

    private int partion(int[] nums, int start, int end) {
        int pivot = start;
        int temp;
        while (start <= end) {
            while (start <= end && nums[start] <= nums[pivot]) start++;
            while (start <= end && nums[end] > nums[pivot]) end--;
            if (start > end) break;
            temp = nums[start];
            nums[start] = nums[end];
            nums[end] = temp;
        }
        temp = nums[end];
        nums[end] = nums[pivot];
        nums[pivot] = temp;
        return end;
    }

    /**
     For example, given array S = [1, 0, -1, 0, -2, 2], and target = 0.
     A solution set is:
     [
     [-1,  0, 0, 1],
     [-2, -1, 1, 2],
     [-2,  0, 0, 2]
     ]
     */
    public List<List<Integer>> fourSum(int[] nums, int target) {
        Set<List<Integer>> set = new HashSet<>();
        List<List<Integer>> res = new ArrayList<>();
        if(nums.length<4) return res;
        Arrays.sort(nums);
        for (int i = 0; i < nums.length; i++) {
            for (int j = i+1; j < nums.length; j++) {
                int low = j+1;
                int high = nums.length-1;
                while(low<high){
                    int sum = nums[i] + nums[j] + nums[low] + nums[high];
                    if(sum ==target){
                        List<Integer> tmp = Arrays.asList(nums[i], nums[j], nums[low], nums[high]);
                        if(!set.contains(tmp)) {
                            res.add(tmp);
                        }
                        set.add(tmp);
                        while(low<nums.length-1 && nums[low]==nums[low+1])low++;
                        while(high>0 && nums[high]==nums[high-1])high--;
                        high--;
                        low++;
                    }else if(sum<target) low++;
                    else high--;
                }
            }
        }
        return res;
    }

    public boolean canPartition2(int[] nums) {
        if (nums == null || nums.length == 0) {
            return true;
        }
        int volumn = 0;
        for (int num : nums) {
            volumn += num;
        }
        if (volumn % 2 != 0) {
            return false;
        }
        volumn /= 2;
        // dp def
        boolean[] dp = new boolean[volumn + 1];
        // dp init
        dp[0] = true;
        // dp transition
        for (int i = 1; i <= nums.length; i++) {
            for (int j = volumn - nums[i-1]; j >=0 ; j--) {
                if(dp[j])  dp[j+nums[i-1]] = true;
            }
        }
        return dp[volumn];
    }

    public boolean canPartition(int[] nums) {
        if(nums.length<2) return false;
        int sum = Arrays.stream(nums).sum();
        if(sum%2!=0) return false;
        int target = sum/2;
        return canPartitionReq(nums,0,0,0,target);
    }


    private boolean canPartitionReq(int[] nums, int i, int sum1, int sum2, int target) {
        if(sum1>target || sum2 > target) return false;
        if(sum1==target && sum2==target){
            return true;
        }
        return canPartitionReq(nums,i+1,sum1+nums[i],sum2,target) ||canPartitionReq(nums,i+1,sum1,sum2+nums[i],target);
    }

    public boolean canPartitionKSubsets(int[] nums, int k) {
        if(nums.length<k) return false;
        int sum = Arrays.stream(nums).sum();
        if(sum%k!=0) return false;
        int target = sum/k;
        int[] groups = new int[k];
        Arrays.sort(nums);
        return canPartitionKSubsetsReq(nums,nums.length-1, groups, target);
    }



    private boolean canPartitionKSubsetsReq(int[] nums, int i, int[] groups, int K) {
        if(i<0) {
            return true;
        }
        for (int j = 0; j < groups.length; j++) {
            if(groups[j] + nums[i]>K) continue;
            groups[j]+=nums[i];
            if(canPartitionKSubsetsReq(nums,i-1,groups,K)) return true;
            groups[j]-=nums[i];
        }
        return false;
    }

    public int numSubarrayProductLessThanK(int[] nums, int k) {
        if(k==0) return 0;
        int left = 0;
        int res = 0;
        int product = 1;
        for (int i = 0; i < nums.length; i++) {
            product*=nums[i];
            while(product>=k && left<=i){
                product/=nums[left];
                left++;
            }
            res+= i - left+1;
        }
        return res;
    }

    public int countBattleships(char[][] board) {
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {

            }
        }
        return 0;
    }

    public int threeSumClosest(int[] nums, int target) {
        int sum = nums[0] + nums[1] + nums[2];
        Arrays.sort(nums);
        int i = -1;
        while(i<nums.length-1){
            i++;
            if(i>0 && nums[i]==nums[i-1]) continue;
            int j = i+1;
            int k = nums.length-1;
            while(j<k){
                int tmpSum = nums[i] + nums[j] + nums[k];
                if(Math.abs(tmpSum-target) < Math.abs(sum-target)) {
                    sum = tmpSum;
                }
                if(tmpSum<target){
                    j++;
                }else{
                    k--;
                }
            }
        }
        return sum;
    }

    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        if(nums.length<3) return res;
        Arrays.sort(nums);
        int i=-1;
        while(i<nums.length-2){
            i++;
            if(i>0 && nums[i]==nums[i-1]) {
                continue;
            }
            int j=i+1;
            int k = nums.length-1;
            while(j<k){
                int tmpSum = nums[i] + nums[j] + nums[k];
                if(tmpSum==0){
                    res.add(Arrays.asList(nums[i],nums[j],nums[k]));
                    while(j<k && nums[j]==nums[j+1]){
                        j++;
                    }
                    while(k>j && nums[k]==nums[k-1]){
                        k--;
                    }
                    j++;k--;
                }else if(tmpSum<0){
                    j++;
                }else{
                    k--;
                }
            }

        }
        return res;
    }

    public int pathSum(TreeNode root, int sum) {
        if(root==null) return 0;
        int res = 0;
        if(root.val==sum){
            res++;
        }
        return res + pathSum(root.left, sum - root.val) + pathSum(root.right, sum - root.val);
    }

    public int distributeCandies(int[] candies) {
        Set<Integer> l = new HashSet();
        for (int i = 0; i < candies.length; i++) {
            l.add(candies[i]);
        }
        return Math.min(candies.length/2, l.size());
    }


    public int findPairs(int[] nums, int k) {
        HashMap<Integer,Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            map.put(nums[i],map.getOrDefault(nums[i],0)+1);
        }
        int res = 0;
        for (int i = 0; i < nums.length || !map.isEmpty(); i++) {
            int abs = Math.abs(nums[i] - k);
            map.put(nums[i],map.getOrDefault(nums[i],0)-1);
            if(map.get(nums[i])<=0) map.remove(nums[i]);

            if(map.containsKey(abs)){
                res++;
                map.put(abs,map.get(abs)-1);
                if(map.get(abs)==0) map.remove(abs);
            }
        }
        return res;
    }

    public int[][] matrixReshape(int[][] nums, int r, int c) {
        if(nums.length==0) return nums;
        int ir = nums.length;
        int ic = nums[0].length;
        if(ir*ic != r*c) return nums;
        int[][] res = new int[r][c];
        int oi = 0;
        int oj = 0;
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                res[i][j] = nums[oi][oj++];
                if(oj==ic){
                    oj = 0;
                    oi++;
                }
            }
        }
        return res;
    }


    public int triangleNumber(int[] nums) {
        int res = 0;
        Arrays.sort(nums);
        for (int i = 0; i < nums.length-2; i++) {
            int k = i+2;
            for (int j = i+1; j < nums.length - 1; j++) {
                while(k<nums.length && nums[i] + nums[j] > nums[k]) k++;
                res+=k-j-1;
            }
        }
        return res;
    }


    public boolean isPerfectSquare(int num) {
        int s = 1;
        int e = num/2+1;
        while(s<=e){
            int mid = s + (e-s)/2;
            Long t = Long.valueOf(mid)*mid;
            if(t==num){
                return true;
            }else if(t<num){
                s = mid+1;
            }else{
                e = mid-1;
            }
        }
        return false;
    }

    public boolean judgeSquareSum(int c) {
        int i = 0;
        int j = (int) Math.sqrt(c);
        while(i<=j){
            int res = i*i + j*j;
            if(res < c){
                i++;
            }else if(res>c){
                j--;
            }else{
                return true;
            }
        }
        return false;
    }




    public boolean pattern132(int[] nums) {
        if(nums.length<3) return false;
        int min = Integer.MAX_VALUE;
        for (int i = 0; i < nums.length; i++) {
            min = Math.min(min,nums[i]);
            for (int j = i+1; j < nums.length; j++) {
                if(nums[i]>min && nums[i] < nums[j]){
                    return true;
                }
            }
        }
        return false;
    }

    public List<Integer> findDuplicates2(int[] nums) {
        List<Integer> res = new ArrayList<>();
        for (int i = 0; i < nums.length; ++i) {
            int index = Math.abs(nums[i])-1;
            if (nums[index] < 0)
                res.add(Math.abs(index+1));
            nums[index] = -nums[index];
        }
        return res;
    }

    public List<List<Integer>> subsetsWithDup(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        subsetsWithDupHelper(nums, res, 0, new ArrayList<Integer>());
        return res;
    }

    private void subsetsWithDupHelper(int[] nums, List<List<Integer>> res,int start,List<Integer> temp) {
        res.add(new ArrayList<>(temp));
        for (int i = start; i < nums.length; i++) {
            if(i>start && nums[i]==nums[i-1]) {
                continue;
            }
            temp.add(nums[i]);
            subsetsWithDupHelper(nums, res, i + 1, temp);
            temp.remove(Integer.valueOf(nums[i]));
        }
    }

    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        if(obstacleGrid.length==0 || obstacleGrid[0][0]==1) return 0;
        int m = obstacleGrid.length;
        int n = obstacleGrid[0].length;
        int[][] dp = new int[m][n];
        dp[0][0]=1;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if(obstacleGrid[i][j]==1){
                    continue;
                }else if(i==0 ){
                    if(j>0 && dp[i][j-1]==0){
                        continue;
                    }
                    dp[i][j] = 1;
                }else if(j==0 ) {
                    if(i>0 && dp[i-1][j]==0){
                        continue;
                    }
                    dp[i][j] = 1;
                }else {
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
                }
            }
        }
        return dp[m-1][n-1];
    }



    //allow no more than 2 duplicates
    public int removeDuplicates(int[] nums) {
        if(nums.length==0) return 0;
        int s = 0;
        int last = nums[0];
        boolean allow = true;

        for (int i = 1; i < nums.length; i++) {
            if(nums[i]!=nums[s] || allow){
                if(allow && last==nums[i]) {
                    allow = false;
                }else{
                    allow = true;
                }
                s++;
                nums[s] = nums[i];
                last = nums[s];
            }
        }
        return s+1;
    }


    public int[] findRightInterval(Interval[] intervals) {
        Map<Interval,Integer> map = new HashMap<>();
        for (int i = 0; i < intervals.length; i++) {
            map.put(intervals[i],i);
        }
        Arrays.sort(intervals, new Comparator<Interval>() {
            @Override
            public int compare(Interval interval, Interval t1) {
                return Integer.compare(interval.start,t1.start);
            }
        });
        int[] res = new int[intervals.length];
        for (int i = 0; i < intervals.length; i++) {
            Interval nextRight = intervalBinarySearch(intervals, 0, intervals.length - 1, intervals[i].end);
            res[map.get(intervals[i])] = nextRight==null? -1 :map.get(nextRight);
        }
        return res;
    }

    private Interval intervalBinarySearch(Interval[] intervals, int s, int e, int target) {
        if(s>e) return null;
        int mid = s+(e-s)/2;
        if(intervals[mid].start>=target){
            if(mid==0 || intervals[mid-1].start<target) {
                return intervals[mid];
            }else{
                return intervalBinarySearch(intervals,s,mid-1,target);
            }
        }else if(intervals[mid].start>target){
            return intervalBinarySearch(intervals,s,mid-1,target);
        }else{
            return intervalBinarySearch(intervals,mid+1,e,target);
        }
    }




    class Coordinate{
        int x;
        int y;
        int dist = 1;
        public Coordinate(int x,int y){
            this.x=x;
            this.y=y;
        }
        public Coordinate(int x,int y,int dist){
            this.x=x;
            this.y=y;
            this.dist=dist;
        }
    }

    public int findMedian(int[] nums) {
        int mid = quickSelect(nums,0,nums.length-1,nums.length/2);
        return mid;
    }

    private int quickSelect(int[] nums, int start, int end, int target) {
        if(start==end) return nums[start];
        int pivot = partition(nums, start, end);
        if(pivot==target){
            return nums[pivot];
        }else if(target<pivot){
            return quickSelect(nums,start,pivot-1,target);
        }else{
            return quickSelect(nums,pivot+1,end,target);
        }
    }



    private int partition(int[] nums, int start, int end) {
        int pivot = start;
        while(start<=end){
            if(start<=end && nums[start]<=nums[pivot]) start++;
            if(start<=end && nums[end]>nums[pivot]) end--;
            if(start>end){
                break;
            }
            int tmp = nums[start];
            nums[start] = nums[end];
            nums[end] = tmp;
        }
        int tmp = nums[end];
        nums[end] = nums[pivot];
        nums[pivot] = tmp;
        return end;
    }

    public int[] nextGreaterElement(int[] findNums, int[] nums) {
        Stack<Integer> stack = new Stack<>();
        Map<Integer,Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            while(!stack.isEmpty() && stack.peek()<nums[i]){
                map.put(stack.pop(),nums[i]);
            }
            stack.add(nums[i]);
        }
        while(!stack.isEmpty()){
            map.put(stack.pop(),-1);
        }
        int[] res = new int[findNums.length];
        for (int i = 0; i < findNums.length; i++) {
            res[i] = map.get(findNums[i]);
        }
        return res;
    }

    /**
     Input: [1,2,1]
     Output: [2,-1,2]
     */
    public int[] nextGreaterElements2(int[] nums) {
        Stack<Integer> stack = new Stack<>();
        Map<Integer,Integer> map = new HashMap<>();
        for (int i = 0; i <= nums.length; i++) {
            for (int j = 0; j < nums.length; j++) {
                if (!stack.isEmpty() && stack.peek() < nums[(i+j+1) % nums.length]) {
                    map.put(stack.pop(), nums[(i+j+1)%nums.length]);
                    break;
                }
            }
            if(i<nums.length) {
                stack.add(nums[i]);
            }
        }

        while(!stack.isEmpty()){
            map.put(stack.pop(),-1);
        }

        int[] res = new int[nums.length];
        for (int i = 0; i < nums.length; i++) {
            res[i] = map.get(nums[i]);
        }
        return res;
    }


    public List<List<Integer>> updateMatrix(List<List<Integer>> matrix) {
        if(matrix.size()==0) return matrix;

        for (int i = 0; i < matrix.size(); i++) {
            for (int j = 0; j < matrix.get(0).size(); j++) {
                int num = matrix.get(i).get(j);
                if(num!=0) {
                    bfsMatrix(matrix, i, j);
                }
            }
        }
        for (List<Integer> list : matrix) {
            for (Integer integer : list) {
                System.out.print(integer+" ");
            }
            System.out.println();
        }
        return matrix;
    }

    private void bfsMatrix(List<List<Integer>> matrix, int i,int j) {
        List<Coordinate> shift = Arrays.asList(new Coordinate(i,1+j),new Coordinate(1+i,j),new Coordinate(i,-1+j),new Coordinate(-1+i,j));
        Queue<Coordinate> queue = new LinkedList<>(shift);
        while(!queue.isEmpty()){
            Coordinate poll = queue.poll();
            if(poll.x>=0 && poll.x<matrix.size() && poll.y>=0 && poll.y<matrix.get(0).size()){
                if(matrix.get(poll.x).get(poll.y)!=0) {
                    int newDist = poll.dist+1;
                    queue.addAll(Arrays.asList(new Coordinate(poll.x, 1 + poll.y,newDist), new Coordinate(1 + poll.x, poll.y,newDist),
                            new Coordinate(poll.x, -1 + poll.y,newDist), new Coordinate(-1 + poll.x, poll.y,newDist)));
                }else {
                    matrix.get(i).set(j, poll.dist);
                    return;
                }
            }
        }

    }

    public int longestSubstringWithoutRepeatedCharacters(String s) {
        return 0;
    }

    public String longestPalindrome(String s) {
        return "";
    }

    //    dp[i][j]: the longest palindromic subsequence's length of substring(i, j)
//    State transition:
//    dp[i][j] = dp[i+1][j-1] + 2 if s.charAt(i) == s.charAt(j)
//    otherwise, dp[i][j] = Math.max(dp[i+1][j], dp[i][j-1])
//    Initialization: dp[i][i] = 1
    public int longestPalindromeSubseq(String s) {
        int[][] dp = new int[s.length()][s.length()];

        for (int i = s.length() - 1; i >= 0; i--) {
            dp[i][i] = 1;
            for (int j = i+1; j < s.length(); j++) {
                if (s.charAt(i) == s.charAt(j)) {
                    dp[i][j] = dp[i+1][j-1] + 2;
                } else {
                    dp[i][j] = Math.max(dp[i+1][j], dp[i][j-1]);
                }
            }
        }
        return dp[0][s.length()-1];
    }

    public int longestPalindromeSubseqMemoriz(String s) {
        return helperLongestPalindrom(s, 0, s.length() - 1);
    }

    private int helperLongestPalindrom(String s, int i, int j) {
        if(i==j) return 1;
        if(s.charAt(i)==s.charAt(j) && i+1==j){
            return 2;
        }
        if(s.charAt(i)==s.charAt(j)){
            return helperLongestPalindrom(s,i+1,j-1) +2;
        }
        return Math.max(helperLongestPalindrom(s, i + 1, j), helperLongestPalindrom(s, i, j - 1));
    }

    public List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        Arrays.sort(nums);
        if(nums.length==1) {
            res.add(Arrays.asList(nums[0]));
            return res;
        }
        boolean[] used = new boolean[nums.length];
        combinationSumHelper(nums, res, new ArrayList<>(),used);
        return res;
    }

    private void combinationSumHelper(int[] candidates,  List<List<Integer>> res,List<Integer> tmp,boolean[] used) {
        if(tmp.size()==candidates.length){
            res.add(new ArrayList<>(tmp));
            return;
        }
        for (int i = 0; i < candidates.length; i++) {
            if(used[i] || (i>0 && candidates[i]==candidates[i-1] && !used[i-1])) continue;
            tmp.add(candidates[i]);
            used[i] = true;
            combinationSumHelper(candidates, res, tmp, used);
            tmp.remove(Integer.valueOf(candidates[i]));
            used[i] = false;
        }
    }

    public List<String> readBinaryWatch(int num) {
        List<String> hours = Arrays.asList("01","02","04","08");
        List<String> min = Arrays.asList("01","02","04","08","16","32");
        List<String>  res = new ArrayList<>();
        for (int i = 0; i <= num; i++) {
            helperBinaryWatch(i, num - i, res,hours,min);
        }
        return res;
    }

    private void helperBinaryWatch(int h, int m, List<String> res, List<String> hours,List<String> min) {

        Set<Integer> hoursSet = new HashSet<>();
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < hours.size(); j++) {
                if(j!=i) {
                }
            }
        }
    }


    //or K-common substring
    public String longestCommonSubstring(String s,String t) {
        int[][] dp = new int[s.length()][t.length()];
        int start = 0;
        int end = 0;
        for (int i = 0; i < s.length(); i++) {
            for (int j = 0; j < t.length(); j++) {
                if(i==0 || j==0) {
                    if (s.charAt(i) == t.charAt(j)){
                        dp[i][j] = 1;
                        start=j;
                    }
                }else{
                    dp[i][j] = dp[i-1][j-1] + (s.charAt(i) == t.charAt(j) ? 1 :0);
                    if(dp[i-1][j-1]+1==dp[i][j]){
                        end= j;
                    }
                }
            }
        }
        printMatrix(dp);
        return t.substring(start,end+1);
    }

    public String minWindow(String s, String t) {
        if(s.isEmpty() || t.isEmpty()) return "";
        int[] hash = new int[256];
        for (int i = 0; i < t.length(); i++) {
            hash[t.charAt(i)]++;
        }
        int left = 0;
        int right = 0;
        int count = t.length();
        int minStart = 0;
        int minEnd = s.length();
        int min = Integer.MAX_VALUE;
        while(right<s.length()){
           if(hash[s.charAt(right)]>0){
               count--;
           }
           hash[s.charAt(right)]--;
           right++;
            while(count==0){
                if(right-left<min && right-left>=t.length()){
                    minStart = left;
                    minEnd = right;
                    min = right-left;
                }
                    if(hash[s.charAt(left)]>=0){
                        count++;
                    }
                    hash[s.charAt(left)]++;
                    left++;
            }
        }
        return min==Integer.MAX_VALUE?"":s.substring(minStart,minEnd);
    }

    public List<Integer> findAnagrams(String s, String p) {
        List<Integer> res = new ArrayList<>();
        if(s.isEmpty() || p.isEmpty()) return res;
        int[] hash = new int[256];
        for (int i = 0; i < p.length(); i++) {
            hash[p.charAt(i)]++;
        }
        int left = 0;
        int right = 0;
        int count = p.length();

        while(right<s.length()){
            if(hash[s.charAt(right)]>0){
                count--;
            }
            hash[s.charAt(right)]--;
            right++;

            if(count==0) {
                res.add(left);
            }

            if(right-left==p.length()){
                if(hash[s.charAt(left)]>=0){
                    count++;
                }
                hash[s.charAt(left)]++;
                left++;
            }
        }
        return res;
    }

    public int wiggleMaxLength(int[] nums) {
        if(nums.length<2) return nums.length;
        int up = 0;
        int down = 0;

        for (int i = 1; i < nums.length; i++) {
            if(nums[i]>nums[i-1]){
                up = down+1;
            }else if(nums[i]<nums[i-1]){
                down = up+1;
            }
        }
        return Math.max(up,down)+1;
    }

     public static class Interval {
             int start;
             int end;
             Interval() { start = 0; end = 0; }
             Interval(int s, int e) { start = s; end = e; }
         }
    public static class NodeWithDepth {
        TreeNode node;
        int depth;
        NodeWithDepth(TreeNode node, int d) {
            this.node= node;
            this.depth = d;
        }
    }


    public List<Integer> largestValues(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        helper(root,result,0);
        return result;
    }

    private void helper(TreeNode root, List<Integer> result, int d) {
        if(root==null) return;

        if(d==result.size()){
            result.add(root.val);
        }else{
            result.set(d,Math.max(root.val,result.get(d)));
        }
        helper(root.left,result,d+1);
        helper(root.right,result,d+1);
    }

    public int findMinArrowShots(int[][] points) {
        if(points.length<=1) return points.length;
        int res = 1;
        Arrays.sort(points, (ints, t1) -> Integer.compare(ints[1],t1[1]));
        int end = points[0][1];
        for (int i = 1; i < points.length; i++) {
            if(points[i][0]>end){
                res++;
                end = points[i][1];
            }
        }
        return res;
    }

    public int eraseOverlapIntervals(Interval[] intervals) {
        if(intervals.length<=1) return 0;
        Arrays.sort(intervals, (interval, t1) -> Integer.compare(interval.end,t1.end));
        int res = 0;
        int e = intervals[0].end;
        for (int i = 1; i < intervals.length; i++) {
            if(intervals[i].start<e) res++;
            else e = intervals[i].end;
        }
        return res;
    }


    public int canCompleteCircuit(int[] gas, int[] cost) {
        int possible = 0;
        for (int i = 0; i < gas.length; i++) {
            possible+=gas[i] - cost[i];
        }
        if(possible<0) return -1;
        int start = 0;
        int accomulate = 0;
        for (int i = 0; i < gas.length; i++) {
            accomulate+= gas[i] - cost[i];
            if(accomulate<0){
                start = i+1;
                accomulate=0;
            }
        }
        return start;
    }



    public int search(int[] nums,int target) {
        if(nums.length==0) return -1;
        int s = 0;
        int e = nums.length-1;
        while(s<e){
            int mid = s+(e-s)/2;
            if(nums[mid]==target) return mid;

            if(nums[mid]>=nums[s]){
                if(target<nums[mid] && target>=nums[s]){
                    e = mid-1;
                }else{
                    s = mid+1;
                }
            }else{
                if(target>nums[mid] && target<=nums[e]){
                    s = mid+1;
                }else{
                    e = mid-1;
                }
            }
        }
        return nums[s]==target?s:-1;
    }

    //2, 3, 6, 7}, 7
    public int combinationSumNum(int[] candidates, int target) {
       int[] dp = new int[target+1];
        for (int i = 1; i <= target; i++) {
            for (int j = 0; j < candidates.length; j++) {
                if(i==candidates[j]){
                    dp[i]+=1;
                }
                if(i-candidates[j]>0) {
                    dp[i] += dp[i - candidates[j]];
                }
            }
        }
        return dp[target];
    }




    public List<String> letterCombinations(String digits) {
        LinkedList<String> ans = new LinkedList<String>();
        if(digits.isEmpty()) return ans;
        ans.add("");
        String[] mapping = new String[] {"0", "1", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        for (int i = 0; i < digits.length(); i++) {
            int j = Character.getNumericValue(digits.charAt(i));
            while(ans.peek().length()==i){
                String t = ans.remove();
                for (int k = 0; k < mapping[j].length(); k++) {
                    ans.add(t+mapping[j].charAt(k));
                }
            }
        }
        return ans;
    }


    public int waysToTraverseGrid(int[][] grid){
        int[][] dp = new int[grid.length][grid[0].length];
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if(grid[i][j]!=0){
                    dp[i][i]=0;
                }else {
                    if (i == 0 || j == 0) dp[i][j] = 1;
                    else dp[i][j] = dp[i-1][j]+ dp[i][j-1];
                }
            }
        }
        for (int j=0;j< grid.length;j++) {
            for (int i = 0; i < grid[0].length ; i++) {
                System.out.print(dp[j][i] + " ");
            }
            System.out.println();
        }
        return dp[grid.length-1][grid[0].length-1];
    }

    public int countNumberOfScores(int N, int[] ar){
        int[] dp1 = new int[N+1];
        for (int i = 1; i <= N; i++) {
            for (int j = 0; j < ar.length; j++) {
                if(ar[j]<i){
                    dp1[i]+= dp1[i-ar[j]];
                }
                if(ar[j]==i){
                    dp1[i]+=1;
                }
            }
        }
        for (int i = 0; i < dp1.length; i++) {
            System.out.println(i+" "+dp1[i]);
        }

        int[][] dp = new int[ar.length][N+1];
        for (int i = 0; i < ar.length; i++) {
            dp[i][0]=1;
            for (int j = 1; j <=N ; j++) {
                int withPlay = j>=ar[i] ?dp[i][j-ar[i]]:0;
                int withoutPlay = i>0?dp[i-1][j]:0;
                dp[i][j] = withoutPlay+withPlay;
            }
        }
        for (int j=0;j< ar.length;j++) {
            for (int i = 0; i <=N ; i++) {
                System.out.print(dp[j][i] + " ");
            }
            System.out.println();
        }
        return dp[ar.length-1][N];
    }

    public static char[][] getACharGrid(){
        char[][] ar = new char[4][5];
        ar[0] = new char[]{1,1,0,0,0};
        ar[1] = new char[]{1,1,0,0,0};
        ar[2] = new char[]{0,0,1,0,0};
        ar[3] = new char[]{0,0,0,1,1};
        return ar;
    }

    public static ListNode getSortedList(){
        ListNode node = new ListNode(1);
        node.next = new ListNode(2);
        node.next.next = new ListNode(3);
        node.next.next.next = new ListNode(4);
        node.next.next.next.next = new ListNode(5);
        node.next.next.next.next.next = new ListNode(6);
        node.next.next.next.next.next.next= new ListNode(7);
        return node;
    }

    public static void printMatrix(int[][] m){
        for (int i = 0; i < m.length; i++) {
            for (int j = 0; j < m[0].length; j++) {
                System.out.print(m[i][j] + " ");
            }
            System.out.println();
        }
    }
    public static TreeNode getTree(){
        TreeNode node = new TreeNode(5);
        node.left = new TreeNode(3);
        node.left.left = new TreeNode(2);
        node.left.right = new TreeNode(4);
        node.left.left.left = new TreeNode(1);

        node.right = new TreeNode(10);
        node.right.right = new TreeNode(100);
        return node;
    }

    public static class ListNode {
             int val;
             ListNode next;
             ListNode(int x) { val = x; }

        static ListNode build(int[] ar){
            ListNode first = new ListNode(ar[0]);
            ListNode current = first;
            for (int i = 1; i < ar.length; i++) {
                current.next = new ListNode(ar[i]);
                current = current.next;
            }
            return first;
        }
    }

    public static class TreeNode {
             int val;
             TreeNode left;
             TreeNode right;
             TreeNode(int x) { val = x; }
         }

    public int numberOfwaysToFormNumber(int[] nums,int n) {
        // dp def
        int[] dp = new int[n + 1];
        // dp init
        dp[0] = 1;
        // dp transition
        for (int i = 1; i <= nums.length; i++) {
            for (int j = 0; j <nums.length-nums[i-1] ; j++) {
                dp[j] += dp[j+nums[i-1]];
            }
        }
        return dp[n];
    }





    public int findDuplicate(int[] nums) {
        Arrays.sort(nums);
        return findBinary(nums,0,nums.length-1);
    }

    private int findBinary(int[] nums, int s, int e) {
        int mid = s+ (e-s)/2;
        if(nums[mid]==nums[mid-1]){
            return nums[mid-1];
        }else {
            int res = findBinary(nums, s, mid);
            if(res==-1){
                res = findBinary(nums,mid+1,e);
            }
            return res;
        }
    }


    public int missingNumber(int[] nums) {
        Arrays.sort(nums);
        int left = 0, right = nums.length;
        int mid;
        while(left<right){
            mid = (left + right)/2;
            if(nums[mid]>mid) right = mid;
            else left = mid+1;
        }
        return left;
    }


    public void moveZeroes(int[] nums) {
        int n = 0;
        int z = 0;
        while(true){
            while(n<nums.length && nums[n]==0) n++;
            while(z<nums.length && nums[z]!=0) z++;
            if(n==nums.length || z==nums.length) break;
            if(n>z) {
                int tmp = nums[n];
                nums[n] = nums[z];
                nums[z] = tmp;
            }else{
                n++;
            }
        }
        System.out.println();
    }

    private int knapsack(int W, int[] weights, int[] val) {
        int[][] dp = new int[val.length+1][W+1];

        for (int i = 0; i <= weights.length; i++) {

        }

        return dp[val.length+1][W];
    }

    public int coinChange(int[] coins, int amount) {
        if(amount==0) return 0;
        if(coins.length==0) return -1;
        int[] dp = new int[amount+1];
        for (int i = 1; i <= amount; i++) {
            dp[i] = Integer.MAX_VALUE;
                for (int j=0;j<coins.length;j++){
                    if(coins[j]==i){
                        dp[i] = 1;
                    }else if(i-coins[j]>0 && dp[i - coins[j]]!=Integer.MAX_VALUE) {
                        dp[i] = Math.min(dp[i], dp[i - coins[j]] + 1);
                    }
                }
        }

        return dp[amount]==Integer.MAX_VALUE?-1:dp[amount];
    }

    public boolean isSubsequence(String s, String t) {
        if(s.length()==0) return true;
        if(t.length()==0 && s.length()>0) return false;
        int sCount = 0;
        int tCount = 0;
        while(sCount<s.length() && tCount<t.length()){
            if(t.charAt(tCount)==s.charAt(sCount)){
                sCount++;
            }
            tCount++;
        }
        return sCount==s.length();
    }

    public int lengthOfLIS(int[] nums) {
        if(nums.length==0) return 0;
        int[] dp = new int[nums.length];
        dp[0] = 1;
        for (int i = 1; i < nums.length; i++) {
            dp[i] = 1;
            for (int j = 0; j < i; j++) {
                if(nums[i]>nums[j]){
                    dp[i] = Math.max(dp[j]+1,dp[i]);
                }
            }
        }
        int res = 0;
        for (int i = 0; i < dp.length; i++) {
            if(dp[i]>res){
                res = dp[i];
            }
        }
        return res;
    }

    public int integerBreak(int n) {
        int[] dp = new int[n+1];
        for (int i = 1; i < n; i++) {
            dp[i] = i;
        }
        for (int i = 2; i <= n; i++) {
            for (int j = 1; j <= i-j; j++) {
                dp[i] = Math.max(dp[j]*dp[i-j],dp[i]);
            }
        }
        return dp[n];
    }

    public int climbStairs2(int n) {
        if(n<3) return n;
        int[] dp = new int[n+1];
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            dp[i] = dp[i-1]+ dp[i-2];
        }
        return dp[n];
    }

    public int climbStairs(int n) {
        int[] memo = new int[n+1];
        return climbStairsHelper(n,0,memo);
    }

    private int climbStairsHelper(int n, int i, int[] memo) {
        if(i>n) return 0;
        if(n==i) return 1;
        if(memo[i]>0) return memo[i];

        int res = climbStairsHelper(n, i + 1, memo) + climbStairsHelper(n, i + 2, memo);
        memo[i] = res;
        return res;
    }

    public int rob(int[] nums) {
        int[][] dp = new int[nums.length+1][2];
        for (int i = 1; i <= nums.length; i++) {
            dp[i][0] = Math.max(dp[i-1][1],dp[i-1][0]);
            dp[i][1] = Math.max(dp[i-1][0]+nums[i-1],nums[i-1]);
        }
        return Math.max(dp[nums.length][0], dp[nums.length][1]);
    }

    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> list = new ArrayList<>();
        if(root==null) return list;
        Stack<TreeNode> stack = new Stack<>();
        while (!stack.isEmpty() || root!=null){
            while(root!=null){
                stack.push(root);
                root = root.left;
            }
            TreeNode pop = stack.pop();
            list.add(pop.val);
            root = pop.right;
        }
        return list;
    }

    public boolean isValidBSTIteratively(TreeNode root) {
        if(root==null) return true;
        Stack<TreeNode> stack = new Stack<>();
        Integer pre = null;
        while (!stack.isEmpty() || root!=null){
            while(root!=null){
                stack.push(root);
                root = root.left;
            }
            TreeNode pop = stack.pop();
            if(pre!=null && pre>=pop.val){
                return false;
            }
            pre = pop.val;
            root = pop.right;
        }
        return true;
    }

    public boolean isValidBST(TreeNode root) {
        if(root==null) return true;
        if(root.left==null && root.right==null) return true;
        return isValidBSTReq(root,Integer.MIN_VALUE,Integer.MAX_VALUE);
    }

    private boolean isValidBSTReq(TreeNode node, int min, int max) {
        if(node==null) return true;
        if(node.val<=min || node.val>=max){
            return false;
        }
        return isValidBSTReq(node.left,min,node.val) && isValidBSTReq(node.right,node.val,max);
    }


    public TreeNode sortedListToBSTIterative(ListNode head) {
        if(head==null) return null;
        return sortedListToBSTIterativeReq(head, null);
    }

    public TreeNode sortedListToBSTIterativeReq(ListNode head,ListNode tail) {
        ListNode slow = head;
        ListNode fast = head;
        if(head==tail) return null;
        while(fast!=tail && fast.next !=tail){
            slow = slow.next;
            fast = fast.next.next;
        }
        TreeNode node = new TreeNode(slow.val);
        node.left = sortedListToBSTIterativeReq(head,slow);
        node.right = sortedListToBSTIterativeReq(head.next,tail);
        return node;
    }

    public TreeNode sortedListToBST(ListNode head) {
        ListNode c = head;
        int length = 0;
        while(c!=null) {
            length++;
            c=c.next;
        }
        this.head = head;
        return sortedListToBstReq(0,length);
    }
    private ListNode head;
    private TreeNode sortedListToBstReq(int s, int e) {
        if(s>e || head==null)return null;
        int mid = s + (e-s)/2;
        TreeNode left = sortedListToBstReq(s,mid-1);
        TreeNode root = new TreeNode(head.val);
        root.left = left;
        head = head.next;
        root.right = sortedListToBstReq(mid+1,e);
        return root;
    }

    public TreeNode sortedArrayToBST(int[] nums) {
        return sortedArrayToBstReq(nums,0,nums.length-1);
    }

    private TreeNode sortedArrayToBstReq(int[] nums, int s, int e) {
        if(s>e) return null;
        int mid = s + (e-s)/2;
        TreeNode node = new TreeNode(nums[mid]);
        node.left = sortedArrayToBstReq(nums,s,mid-1);
        node.right = sortedArrayToBstReq(nums,mid+1,e);
        return node;
    }

    public boolean isSameTree(TreeNode p, TreeNode q) {
        if(p==null && q==null){
            return true;
        }
        if(p==null || q==null){
            return false;
        }
        return isSameTree(p.left,q.left) && isSameTree(p.right,q.right) && p.val==q.val;
    }

    public int maxDepth(TreeNode root) {
        if(root==null) return 0;
        return 1 + Math.max(maxDepth(root.left),maxDepth(root.right));
    }

    public int singleNumber2(int[] nums) {
        int c = 0;
        for (int i = 0; i < nums.length; i++) {
            c ^= nums[i];
        }
        return c;
    }


    public int[] intersect(int[] nums1, int[] nums2) {
        Map<Integer, Integer> cache = new HashMap<>();
        int[] smaller = nums1;
        int[] longer = nums2;
        if (nums1.length > nums2.length) {
            smaller = nums2;
            longer = nums1;
        }
        for (int i = 0; i < smaller.length; i++) {
            cache.put(smaller[i], cache.getOrDefault(smaller[i], 0) + 1);
        }
        List<Integer> res = new ArrayList<>();
        for (int i = 0; i < longer.length; i++) {
            if (cache.getOrDefault(longer[i],0)>0) {
                res.add(longer[i]);
                cache.put(longer[i],cache.get(longer[i])-1);
            }
        }

        int[] resA = new int[res.size()];
        int c = 0;
        for (Integer num : res) {
            resA[c++] = num;
        }
        return resA;
    }

    public char findTheDifference(String s, String t) {
        int c = 0;
        for (int i = 0; i < s.length(); ++i) {
            char c1 = s.charAt(i);
            c = c ^ c1;
        }
        return (char) c;
    }

    public int hammingDistance(int x, int y) {
        int xor = x ^ y;
        int res = 0;
        while (xor != 0) {
            res += xor & 1;
        }
        return res;
    }


    public boolean isPalindrome(String s) {
        if (s.isEmpty()) return true;
        s = s.toLowerCase();
        int start = 0;
        int end = s.length() - 1;
        while (start < end) {
            while (start < end && !Character.isLetterOrDigit(s.charAt(start))) {
                start++;
            }
            while (end > start && !Character.isLetterOrDigit(s.charAt(end))) {
                end--;
            }
            if (s.charAt(start) != s.charAt(end)) {
                return false;
            }
            start++;
            end--;
        }
        return start >= end;
    }

    public boolean isValid(String s) {
        Stack<Character> stack = new Stack<>();
        List<Character> open = Arrays.asList('{', '[', '(');
        List<Character> close = Arrays.asList('}', ']', ')');
        for (int i = 0; i < s.length(); i++) {
            char curr = s.charAt(i);
            if (open.contains(curr)) {
                stack.push(curr);
            } else if (close.contains(curr)) {
                if (stack.isEmpty()) return false;
                Character pop = stack.pop();
                if ((pop == '{' && curr != '}') || (pop == '[' && curr != ']') || (pop == '(' && curr != ')')) {
                    return false;
                }
            }
        }
        return stack.isEmpty();
    }


    public boolean find132pattern(int[] nums) {
        if (nums.length < 3) return false;
        int start = 0;
        while (start < nums.length - 1) {
            while (start < nums.length && nums[start] >= nums[start + 1]) start++;

            int next = start + 1;
            while (next < nums.length && nums[next] < nums[next + 1]) next++;

            for (int i = next + 1; i < nums.length; i++) {
                if (nums[i] < nums[next] && nums[i] > nums[start]) {
                    return true;
                }
            }
            start = start + 1;
        }
        return false;
    }


    public int maxProfit(int[] prices) {
        int max = 0;
        for (int i = 1; i < prices.length; i++) {
            max += Math.max(0, prices[i] - prices[i - 1]);
        }
        return max;
    }



    public int findKth(int[] ar, int start, int end, int k) {
        int pivot = ar[end];
        int left = start;
        int right = end;
        while (true) {
            while (ar[left] < pivot && left < right) left++;
            while (ar[right] >= pivot && right > left) right--;
            if (left == right) break;
            swap(ar, left, right);
        }
        swap(ar, end, left);
        if (left + 1 == k) return pivot;
        else if (k < left + 1) {
            return findKth(ar, start, left - 1, k);
        } else {
            return findKth(ar, left + 1, end, k - left);
        }
    }

    private void swap(int[] ar, int left, int right) {
        int tmp = ar[left];
        ar[left] = ar[right];
        ar[right] = tmp;
    }

    public int minMoves(int[] nums) {
        int res = 0;
        Arrays.sort(nums);
        int mid;
        int m = nums.length / 2;
        if (nums.length % 2 == 0) {
            mid = (nums[m] + nums[m - 1]) / 2;
        } else {
            mid = nums[m];
        }
        for (int i = 0; i < nums.length; i++) {
            res += (Math.abs(mid - nums[i]));
        }
        return res;
    }

    public int[] productExceptSelf(int[] nums) {
        int n = nums.length;
        int[] res = new int[n];
        res[0] = 1;
        for (int i = 1; i < n; i++) {
            res[i] = res[i - 1] * nums[i - 1];
        }
        int right = 1;
        for (int i = n - 1; i >= 0; i--) {
            res[i] *= right;
            right *= nums[i];
        }
        return res;
    }

    public int maxProduct(int[] nums) {
        int max = nums[0];
        int localMax;
        int localMin;
        int maxHere = nums[0];
        int minHere = nums[0];
        for (int i = 1; i < nums.length; i++) {
            localMax = Math.max(nums[i], Math.max(maxHere * nums[i], minHere * nums[i]));
            localMin = Math.min(nums[i], Math.min(maxHere * nums[i], minHere * nums[i]));
            max = Math.max(max, localMax);
            minHere = localMin;
            maxHere = localMax;
        }
        return max;
    }

    public List<Integer> countSmaller(int[] nums) {
        Integer[] res = new Integer[nums.length];
        Node root = null;
        for (int i = nums.length - 1; i >= 0; i--) {
            root = insert2(root, nums[i], i, res, 0);
        }
        return Arrays.asList(res);
    }

    private Node insert2(Node root, int val, int i, Integer[] res, int presum) {
        if (root == null) {
            res[i] = presum;
            root = new Node(val, 0);
        } else if (root.val == val) {
            root.dup++;
            res[i] = presum + root.sum;
        } else if (val < root.val) {
            root.sum++;
            root.left = insert2(root.left, val, i, res, presum);
        } else {
            root.right = insert2(root.right, val, i, res, presum + root.dup + root.sum);
        }
        return root;
    }

    private Node insert(int num, Node node, int[] ans, int i, int preSum) {
        if (node == null) {
            node = new Node(num, 0);
            ans[i] = preSum;
        } else if (node.val == num) {
            node.dup++;
            ans[i] = preSum + node.sum;
        } else if (node.val > num) {
            node.sum++;
            node.left = insert(num, node.left, ans, i, preSum);
        } else {
            node.right = insert(num, node.right, ans, i, preSum + node.dup + node.sum);
        }
        return node;
    }

    public int maxEnvelopes(int[][] envelopes) {
        if (envelopes.length == 0) return 0;
        if (envelopes.length == 1) return 1;
        Arrays.sort(envelopes, new Comparator<int[]>() {
            @Override
            public int compare(int[] ints, int[] t1) {
                if (ints[0] == t1[0]) {
                    return ints[1] > t1[1] ? -1 : 1;
                }
                return ints[0] > t1[0] ? 1 : -1;
            }
        });
        int[] dp = new int[envelopes.length];
        dp[0] = 1;
        for (int i = 1; i < envelopes.length; i++) {
            dp[i] = 1;
            for (int j = 0; j < i; j++) {
                if (envelopes[i][1] > envelopes[j][1] && dp[i] < dp[j] + 1) {
                    dp[i] = dp[j] + 1;
                }
            }
        }
        int res = 0;
        for (int i = 0; i < dp.length; i++) {
            if (dp[i] > res) {
                res = dp[i];
            }
        }
        return res;
    }

    public int[][] reconstructQueue(int[][] people) {
        Arrays.sort(people, new Comparator<int[]>() {
            public int compare(int[] a, int[] b) {
                if (b[0] == a[0]) return a[1] > b[1] ? 1 : -1;
                return a[0] > b[0] ? -1 : 1;
            }
        });
        int n = people.length;
        ArrayList<int[]> tmp = new ArrayList<>();
        for (int i = 0; i < n; i++)
            tmp.add(people[i][1], people[i]);

        for (int i = 0; i < tmp.size(); i++) {
            people[i] = tmp.get(i);
        }
        return people;
    }

    static class TaskA {

        public void solve(InputReader in, PrintWriter out) {
            int n = in.nextInt();

        }
    }

    static class InputReader {
        public BufferedReader reader;
        public StringTokenizer tokenizer;

        public InputReader(InputStream stream) {
            reader = new BufferedReader(new InputStreamReader(stream), 32768);
            tokenizer = null;
        }

        public String next() {
            while (tokenizer == null || !tokenizer.hasMoreTokens()) {
                try {
                    tokenizer = new StringTokenizer(reader.readLine());
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
            return tokenizer.nextToken();
        }

        public int nextInt() {
            return Integer.parseInt(next());
        }
    }

    class Pair {
        int val;
        int count;
        Pair(int val, int count) {
            this.val = val;
            this.count = count;
        }
    }

    class Node {
        Node right;
        int val;
        Node left;
        int sum;
        int dup = 1;

        Node(int val, int sum) {
            this.val = val;
            this.sum = sum;
        }
    }
}
