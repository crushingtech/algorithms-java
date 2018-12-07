package com.book;

import java.util.ArrayList;
import java.util.List;
import java.util.Stack;

public class Stacks {
    public static void main(String[] args) {
        Stack<Integer> s = new Stack<>();
        s.add(3);
        s.add(1);
        s.add(5);
        s.add(2);
        new Stacks().sortStack(s);
    }

    private void sortStack(Stack<Integer> s) {
        if(!s.isEmpty()){
            Integer pop = s.pop();
            sortStack(s);
            sortStackInsert(pop,s);
        }
    }

    private void sortStackInsert(Integer pop, Stack<Integer> s) {
        if(s.isEmpty() || s.peek().compareTo(pop)>0){
            s.push(pop);
        }else{
            Integer smaller = s.pop();
            sortStackInsert(pop,s);
            s.push(smaller);
        }
    }


}
