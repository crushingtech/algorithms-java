package com.algorithms.stack;


/**
 * Run: echo ( 1 + ( ( 3 * 5 ) * ( 3 + 1 ) ) )
 */
public class EvaluateExpression {
    public static void main(String[] args) {

        System.out.println(new EvaluateExpression().evaluate("( 1 + ( ( 3 * 5 ) * ( 3 + 1 ) ) )"));
    }

    public int evaluate(String expr) {
        Stack<Character> opd = new StackLinkedList<>();
        Stack<Integer> num = new StackLinkedList();
        for (int i = 0; i < expr.length(); i++) {
            char taken = expr.charAt(i);
            if ('(' == taken || ' ' == taken) {

            } else if ('*' == taken || '+' == taken) {
                opd.push(taken);
            } else if (')' == taken) {
                Character operand = opd.pop();
                if ('+' == operand) {
                    num.push(num.pop() + num.pop());
                } else if ('*' == operand) {
                    num.push(num.pop() * num.pop());
                }
            } else {
                num.push(Character.getNumericValue(taken));
            }
        }
        return num.pop();
    }
}
