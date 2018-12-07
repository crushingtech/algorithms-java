package com.book;

import java.util.*;

/**
 * Created by eshulga on 3/29/17.
 */
public class Trees {
    public static void main(String[] args) {
        TreeNode node1 = new TreeNode(1);
//        node1.left = new TreeNode(3);
//        node1.left.left = new TreeNode(1);
//        node1.right= new TreeNode(10);

        TreeNode node2 = new TreeNode(1);
        node2.right = new TreeNode(1);
        node2.left = new TreeNode(1);

//        System.out.println(new Solution().pathSum(node1,8));
//        System.out.println(new Trees().pathSum2(node1, 11));
//        TreeNode x = new Trees().mergeTrees(node1, node2);

//        System.out.println(new Trees().averageOfLevels(node1));
//        System.out.println(new Trees().findTarget(node1,2));
//        System.out.println(new Trees().constructMaximumBinaryTree(new int[]{3,2,1,6,0,5}));
//        System.out.println(new Trees().lowestCommonAncestorNotBST(node2, new TreeNode(2), new TreeNode(3)).val);
//        System.out.println(new Trees().isValidBST(node2));
//        System.out.println(new Trees().longestUnivaluePath(node2));
        System.out.println(new Trees().constructMaximumBinaryTree2(new int[]{3, 2, 1, 6, 0, 5}));
    }
    //https://leetcode.com/problems/serialize-and-deserialize-binary-tree/description/
    // Encodes a tree to a single string.
    //10(5(2(1)(3)))
    public String serialize(TreeNode root) {
        StringBuilder sb = new StringBuilder();
        sb.append(root.val);
        serializeInner(root.left,sb);
        serializeInner(root.right,sb);
        return sb.toString();
    }

    public void serializeInner(TreeNode root, StringBuilder sb) {
        if(root==null) return;
        sb.append(root.val+"");
        sb.append(root.val+" ");
        serializeInner(root.left,sb);
        serializeInner(root.right,sb);
    }
    //https://leetcode.com/problems/serialize-and-deserialize-binary-tree/description/
    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        String[] split = data.split(" ");
        return null;
    }


    //https://leetcode.com/problems/find-mode-in-binary-search-tree/description/
    public int[] findMode2(TreeNode root) {
        TreeMap<Integer, Integer> resMap = new TreeMap<>(new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return 0;
            }
        });
        findModeHelper(root, resMap);
        Integer mode = resMap.entrySet().stream().max(Map.Entry.comparingByValue()).get().getValue();
        int[] res = new int[resMap.size()];
        int c = 0;
        for (Map.Entry<Integer, Integer> num : resMap.entrySet()) {
            if(num.getValue()==mode) {
                res[c++] = num.getKey();
            }
        }
        return res;
    }

    private void findModeHelper(TreeNode root, Map<Integer, Integer> resSet) {
        if (root == null) return;
        if (root.left != null && root.val == root.left.val) resSet.put(root.val, resSet.getOrDefault(root.val, 0) + 1);
        if (root.right != null && root.val == root.right.val)
            resSet.put(root.val, resSet.getOrDefault(root.val, 0) + 1);
        findModeHelper(root.left, resSet);
        findModeHelper(root.right, resSet);
    }

    //https://leetcode.com/problems/trim-a-binary-search-tree/description/
    public TreeNode trimBST(TreeNode root, int L, int R) {
        if (root == null) return null;
        root.left = trimBST(root.left, L, R);
        root.right = trimBST(root.right, L, R);
        if (root.val > R || root.val < L) {
            if (root.val < L) return root.right;
            if (root.val > R) return root.left;
        }
        return root;
    }

    //https://leetcode.com/problems/maximum-binary-tree/description/
    public TreeNode constructMaximumBinaryTree2(int[] nums) {
        TreeNode treeNode = constructMaximumBinaryTreeHelper(nums, 0, nums.length - 1);
        return treeNode;
    }

    private TreeNode constructMaximumBinaryTreeHelper(int[] nums, int s, int e) {
        if (s > e) return null;
        int max = Integer.MIN_VALUE;
        int maxIndex = Integer.MIN_VALUE;
        for (int i = s; i <= e; i++) {
            if (nums[i] > max) {
                max = nums[i];
                maxIndex = i;
            }
        }
        TreeNode node = new TreeNode(max);
        node.left = constructMaximumBinaryTreeHelper(nums, s, maxIndex - 1);
        node.right = constructMaximumBinaryTreeHelper(nums, maxIndex + 1, e);
        return node;
    }

    //https://leetcode.com/problems/subtree-of-another-tree/description/
    public boolean isSubtree2(TreeNode s, TreeNode t) {
        if (s == null) return false;
        return isEqual(s, t) || isSubtree2(s.left, t) || isSubtree2(s.right, t);
    }

    public boolean isEqual(TreeNode s, TreeNode t) {
        if (s == null && t == null) return true;
        if (s == null || t == null) return false;
        return s.val == t.val && isEqual(s.left, t.left) && isEqual(s.right, t.right);
    }

    //https://leetcode.com/problems/binary-tree-pruning/description/
    public TreeNode pruneTree(TreeNode root) {
        if (root == null) return null;
        root.left = pruneTree(root.left);
        root.right = pruneTree(root.right);
        if (root.left == null && root.right == null && root.val == 0) return null;
        return root;
    }

    //https://leetcode.com/problems/longest-univalue-path/
    public int longestUnivaluePath(TreeNode root) {
        if (root == null) return 0;
        Map<Integer, Integer> map = new HashMap<>();
        int[] c = new int[1];
        longestUnivaluePathInternal(root, map, c);
        return map.values().stream().max(Comparator.naturalOrder()).orElseGet(() -> new Integer(0));
    }

    private void longestUnivaluePathInternal(TreeNode root, Map<Integer, Integer> map, int[] c) {
        if (root == null) return;
        longestUnivaluePathInternal(root.left, map, c);
        longestUnivaluePathInternal(root.right, map, c);

    }

    public boolean isSubtree(TreeNode s, TreeNode t) {
        if (s == null) return false;
        boolean isSubtree = isSubtreeInternal(s, t);
        if (isSubtree) return true;
        return isSubtreeInternal(s.left, t) || isSubtreeInternal(s.right, t);
    }

    private boolean isSubtreeInternal(TreeNode s, TreeNode t) {
        if (s == null && t == null) return true;
        if (s == null || t == null) return false;
        if (s.val != t.val) return false;
        return isSubtreeInternal(s.left, t.left) && isSubtreeInternal(s.right, t.right);
    }

    public boolean isValidBST(TreeNode root) {
        return false;
    }

    //https://leetcode.com/problems/find-mode-in-binary-search-tree/description/
    public int[] findMode(TreeNode root) {
        int[] res = new int[1];
        return res;
    }

    //https://leetcode.com/problems/sum-of-left-leaves/description/
    public int sumOfLeftLeaves(TreeNode root) {
        if (root == null) return 0;
        int[] c = new int[1];
        sumOfLeftLeavesInternal(root, c);
        return c[0];
    }

    private void sumOfLeftLeavesInternal(TreeNode root, int[] c) {
        if (root == null) return;
        if (root.left != null && root.left.left != null && root.left.right != null) {
            c[0] += root.left.val;
        }
        sumOfLeftLeavesInternal(root.left, c);
        sumOfLeftLeavesInternal(root.right, c);
    }

    //https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/description/#_=_
    public TreeNode lowestCommonAncestorNotBST(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || p == root || q == root) return root;
        TreeNode left = lowestCommonAncestorNotBST(root.left, p, q);
        TreeNode right = lowestCommonAncestorNotBST(root.right, p, q);
        if (left != null && right != null) return root;
        return left != null ? left : right;
    }

    //https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/description/#_=_
    public TreeNode lowestCommonAncestorNotBST2(TreeNode root, TreeNode p, TreeNode q) {
        Object[] seen = new Object[3];
        seen[0] = false;
        seen[1] = false;
        lowestCommonAncestorNotBSTInternal(root, p, q, seen);
        return (TreeNode) seen[2];
    }

    private void lowestCommonAncestorNotBSTInternal(TreeNode node, TreeNode p, TreeNode q, Object[] seen) {
        if (node == null || ((boolean) seen[0] && (boolean) seen[1])) return;
        lowestCommonAncestorNotBSTInternal(node.left, p, q, seen);
        lowestCommonAncestorNotBSTInternal(node.right, p, q, seen);

        if (node.val == p.val) seen[0] = true;
        if (node.val == q.val) seen[1] = true;

        if ((boolean) seen[0] && (boolean) seen[1] && seen[2] == null) {
            seen[2] = node;
        }

    }

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null) return null;
        if (root.val > p.val && root.val > q.val) {
            return lowestCommonAncestor(root.left, p, q);
        } else if (root.val < p.val && root.val < q.val)
            return lowestCommonAncestor(root.right, p, q);
        else {
            return root;
        }
    }

    public TreeNode invertTree(TreeNode root) {
        if (root == null) return null;
        TreeNode tmpleft = root.left;
        root.left = invertTree(root.right);
        root.right = invertTree(tmpleft);
        return root;
    }

    public TreeNode mergeTrees2(TreeNode t1, TreeNode t2) {
        if (t1 == null && t2 == null) return null;
        if (t1 == null) {

            return mergeTrees2(t1, t2);
        }
        return null;
    }

    public TreeNode constructMaximumBinaryTree(int[] nums) {
        return constructMaximumBinaryTreeIn(nums, 0, nums.length - 1);
    }

    public TreeNode constructMaximumBinaryTreeIn(int[] nums, int s, int e) {
        if (s > e) return null;
        int max = Integer.MIN_VALUE;
        int index = Integer.MIN_VALUE;
        for (int i = e; i <= e; i++) {
            if (nums[i] > max) {
                max = nums[i];
                index = i;
            }
        }
        TreeNode node = new TreeNode(nums[index]);
        node.left = constructMaximumBinaryTreeIn(nums, s, index - 1);
        node.right = constructMaximumBinaryTreeIn(nums, index + 1, e);
        return node;
    }

    public boolean findTarget(TreeNode root, int k) {
        List<Integer> list = new ArrayList<>();
        inorder(root, list);
        if (list.size() <= 1) return false;
        int s = 0;
        int e = list.size() - 1;
        while (s < e) {
            int sum = list.get(s) + list.get(e);
            if (sum == k) return true;
            else if (sum > k) e--;
            else s++;
        }
        return false;
    }

    private void inorder(TreeNode root, List<Integer> list) {
        if (root == null) return;
        inorder(root.left, list);
        list.add(root.val);
        inorder(root.right, list);
    }


    public List<Double> averageOfLevels(TreeNode root) {
        List<Double> res = new ArrayList<>();
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            int n = queue.size();
            double sum = 0;
            for (int i = 0; i < n; i++) {
                TreeNode poll = queue.poll();
                sum += poll.val;
                if (poll.left != null) queue.add(poll.left);
                if (poll.right != null) queue.add(poll.right);
            }
            res.add(sum / n);
        }
        return res;
    }

    public void flatten(TreeNode root) {
        if (root == null) return;
        TreeNode left = root.left;
        TreeNode right = root.right;
        root.left = null;
        flatten(right);
        flatten(left);

    }


    public TreeNode mergeTrees(TreeNode t1, TreeNode t2) {
        if (t1 == null && t2 == null) return null;
        if (t1 == null) {
            t1 = new TreeNode(t2.val);
        } else if (t2 == null) {
            return t1;
        } else {
            t1.val += t2.val;
        }
        if (t2.left != null) {
            t1.left = mergeTrees(t1.left, t2.left);
        }
        if (t2.right != null) {
            t1.right = mergeTrees(t1.right, t2.right);
        }
        return t1;
    }


    public List<List<Integer>> pathSum2(TreeNode root, int sum) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) return res;
        pathSum2Helper(root, sum, res, new ArrayList<Integer>());
        return res;
    }

    private void pathSum2Helper(TreeNode root, int sum, List<List<Integer>> res, List<Integer> temp) {
        if (root == null) return;
        temp.add(root.val);
        if (root.left == null && root.right == null) {
            if (sum == root.val) {
                res.add(new ArrayList<>(temp));
            }
            return;
        }
        if (root.left != null) {
            pathSum2Helper(root.left, sum - root.val, res, temp);
            temp.remove(temp.size() - 1);
        }
        if (root.right != null) {
            pathSum2Helper(root.right, sum - root.val, res, temp);
            temp.remove(temp.size() - 1);
        }
    }

    public int pathSum(TreeNode root, int sum) {
        if (root == null) return 0;
        return pathSumHelper(root, sum) + pathSum(root.left, sum) + pathSum(root.right, sum);
    }

    private int pathSumHelper(TreeNode root, int sum) {
        if (root == null) return 0;
        int count = 0;
        if (root.val == sum) count++;
        count += pathSumHelper(root.left, sum - root.val);
        count += pathSumHelper(root.right, sum - root.val);
        return count;
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
