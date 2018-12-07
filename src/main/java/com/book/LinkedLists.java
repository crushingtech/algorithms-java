package com.book;

public class LinkedLists {

    public static void main(String[] args) {
        LinkedLists s = new LinkedLists();
        ListNode node = new ListNode(10);
        ListNode cycle = new ListNode(100);
        node.next = new ListNode(1);
        node.next.next = cycle;
        node.next.next.next = new ListNode(3);
        node.next.next.next = new ListNode(4);
        node.next.next.next.next = cycle;


        ListNode node1 = new ListNode(1);
        node1.next = new ListNode(1);
        node1.next.next = new ListNode(2);
//        System.out.println(s.addTwoNumbers(node, node1));
//        System.out.println(s.addTwoNumbers(node, node1));
        System.out.println(new LinkedLists().detectCycle(node));
    }

    //https://leetcode.com/problems/linked-list-cycle-ii/description/
    public ListNode detectCycle(ListNode head) {
        ListNode first = head;
        ListNode slow = head;
        ListNode fast = head.next;
        while(slow!=fast){
            slow = slow.next;
            fast = fast.next.next;
        }

        while(first!=slow){
            first = first.next;
            slow = slow.next;
        }
        return slow;

    }

    public ListNode sortList(ListNode head) {
        return null;
    }

    //https://leetcode.com/problems/add-two-numbers/description/
    //Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
    //Output: 7 -> 0 -> 8
    //Explanation: 342 + 465 = 807.
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        if (l1 == null && l2 == null) return null;
        return addTwoNumbersInternal(l1, l2, 0);
    }

    private ListNode addTwoNumbersInternal(ListNode l1, ListNode l2, int reminder) {
        if (l1 == null && l2 == null && reminder==0) return null;
        int sum = reminder;
        if (l1 != null) sum += l1.val;
        if (l2 != null) sum += l2.val;
        if (sum >= 10) {
            reminder = 1;
            sum -= 10;
        } else {
            reminder = 0;
        }
        ListNode n = new ListNode(sum);

        n.next = addTwoNumbersInternal(l1 != null ? l1.next : null, l2 != null ? l2.next : null, reminder);
        return n;
    }

    //https://leetcode.com/problems/remove-duplicates-from-sorted-list/description/
    //https://leetcode.com/problems/remove-duplicates-from-sorted-list-ii/description/
    public ListNode deleteDuplicates(ListNode head) {
        ListNode dummyFirst = new ListNode(Integer.MAX_VALUE);
        dummyFirst.next = head;
        ListNode current = dummyFirst;
        ListNode previous = dummyFirst;
        boolean isDup = false;
        while (current != null) {
            if (current.next != null && current.val == current.next.val) {
                current.next = current.next.next;
                isDup = true;
            } else {
                if (isDup) {
                    previous.next = current.next;
                    isDup = false;
                } else {
                    previous = current;
                }
                current = current.next;
            }
        }
        return dummyFirst.next;
    }

    public static class ListNode {
        int val;
        ListNode next;

        @Override
        public String toString() {
            return "ListNode{" +
                    "val=" + val +
                    ", next=" + next +
                    '}';
        }

        ListNode(int x) {
            val = x;
        }
    }
}
