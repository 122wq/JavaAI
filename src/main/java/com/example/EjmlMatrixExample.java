package com.example;

import org.ejml.simple.SimpleMatrix;

public class EjmlMatrixExample {
    public static void main(String[] args) {
        SimpleMatrix A = new SimpleMatrix(new double[][]{
                {1, 2, 3},
                {4, 5, 6}
        });

        System.out.println("A:");
        A.print();

        SimpleMatrix B = new SimpleMatrix(new double[][]{
                {7, 8},
                {9, 10},
                {11, 12}
        });

        System.out.println("B:");
        B.print();

        SimpleMatrix C = A.mult(B);
        System.out.println("C = A * B:");
        C.print();

        SimpleMatrix I = SimpleMatrix.identity(3);
        System.out.println("I3:");
        I.print();
    }
}
