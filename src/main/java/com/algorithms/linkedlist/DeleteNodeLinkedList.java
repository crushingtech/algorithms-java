package com.algorithms.linkedlist;

/**
 * Delete a given node in Linked List under given constraints
 * Given a Singly Linked List, write a function to delete a given node. Your function must follow following constraints:
 * 1) It must accept pointer to the start node as first parameter and node to be deleted as second parameter i.e., pointer to head
 * node is not global.
 * 2) It should not return pointer to the head node.
 * 3) It should not accept pointer to pointer to head node.
 * <p>
 * You may assume that the Linked List never becomes empty.
 * <p>
 * Let the function name be deleteNode(). In a straightforward implementation, the function needs to modify head pointer when
 * the node to be deleted is first node. As discussed in previous post, when a function modifies the head pointer,
 * the function must use one of the given approaches, we can’t use any of those approaches here.
 */
public class DeleteNodeLinkedList {
    public static void main(String[] args) {
        Node node = new Node(1);
        node.next = new Node(2);
        node.next.next = new Node(3);
        node.next.next.next = new Node(4);

        Node run = new DeleteNodeLinkedList().run(node, 4);
        while (run != null) {
            System.out.print(run.data + " ");
            run = run.next;
        }
        System.out.println();
    }

    private Node run(Node node, int data) {
        if (node.data == data) {
            return node.next;
        }
        Node current = node;
        while (current.next != null && current.next.data != data) {
            current = current.next;
        }
        if (current.next != null) {
            current.next = current.next.next;
        }
        return node;
    }

    static class Node {
        Node next;
        int data;

        Node(int data) {
            this.data = data;
        }
    }
}
