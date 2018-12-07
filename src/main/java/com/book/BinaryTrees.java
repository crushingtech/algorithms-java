package com.book;


public class BinaryTrees {
    public static void main(String[] args) {
        TreeNode n = new TreeNode(5);
        n.left = new TreeNode(4);
        n.right = new TreeNode(6);
        n.left.left = new TreeNode(2);
        n.right.left = new TreeNode(33);
        n.right.right = new TreeNode(35);

        int[] ints = new BinaryTrees().largestCompletedNode(n, new int[3]);
        System.out.println(ints);
    }

    private int[] largestCompletedNode(TreeNode node, int[] tmp) {
        if(node!=null && node.left==null && node.right==null) return new int[]{1,0,0};
        if (node == null) return new int[]{0, 0, 0};

        int[] left = largestCompletedNode(node.left, tmp);

        int[] right = largestCompletedNode(node.right, tmp);

        boolean completed = left[0] == 1 && right[0] == 1;
        if (completed) {
            int maxSize = left[1] + right[1] + 1;
            return new int[]{1, maxSize, maxSize};
        } else {
            return new int[]{0, 0, Math.max(left[2], right[2])};
        }
    }

    private int[] isBalanced(TreeNode node, int[] tmp) {
        if (node == null) return new int[]{1, -1};

        int[] left = isBalanced(node.left, tmp);
        if (left[0] == 0) return left;

        int[] right = isBalanced(node.right, tmp);
        if (right[0] == 0) return right;

        int balanced = Math.abs(left[1] - right[1]) > 1 ? 0 : 1;
        int hight = Math.max(left[1], right[1]);
        return new int[]{balanced, hight};

    }

    public static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
    }
}
