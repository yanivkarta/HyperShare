#ifndef __TEST1_H__
#define __TEST1_H__
#include "../include/matrix.h"
//google test:
#include <gtest/gtest.h>

//test matrix
//add unit tests for isolation matrix

using namespace std;
using namespace provallo;
//Basic tests (1,2,3) . 
//more complex tests (4,5,6)
//performance tests (7,8,9)


TEST(HyperMatrix, test1) {
    matrix<double> m1(3, 3);
    m1(0, 0) = 1;
    m1(0, 1) = 2;
    m1(0, 2) = 3;
    m1(1, 0) = 4;
    m1(1, 1) = 5;
    m1(1, 2) = 6;
    m1(2, 0) = 7;
    m1(2, 1) = 8;
    m1(2, 2) = 9;
    matrix<double> m2 = m1 * m1;
    EXPECT_EQ(m2(0, 0), 1);
    EXPECT_EQ(m2(0, 1), 4);
    EXPECT_EQ(m2(0, 2), 9);
    EXPECT_EQ(m2(1, 0), 16);
    EXPECT_EQ(m2(1, 1), 25);
    EXPECT_EQ(m2(1, 2), 36);
    EXPECT_EQ(m2(2, 0), 49);
    EXPECT_EQ(m2(2, 1), 64);
    EXPECT_EQ(m2(2, 2), 81);
}

//test matrix
TEST(HyperMatrix, test2) {
    matrix<double> m1(3, 3);
    m1(0, 0) = 1;
    m1(0, 1) = 2;
    m1(0, 2) = 3;
    m1(1, 0) = 4;
    m1(1, 1) = 5;
    m1(1, 2) = 6;
    m1(2, 0) = 7;
    m1(2, 1) = 8;
    m1(2, 2) = 9;
    matrix<double> m2 = m1 * m1;
    EXPECT_EQ(m2(0, 0), 1);
    EXPECT_EQ(m2(0, 1), 4);
    EXPECT_EQ(m2(0, 2), 9);
    EXPECT_EQ(m2(1, 0), 16);
    EXPECT_EQ(m2(1, 1), 25);
    
}

TEST(HyperMatrix, test3) {
    matrix<double> m1(3, 3);
    m1(0, 0) = 1;
    m1(0, 1) = 2;
    m1(0, 2) = 3;
    m1(1, 0) = 4;
    m1(1, 1) = 5;
    m1(1, 2) = 6;
    m1(2, 0) = 7;
    m1(2, 1) = 8;
    m1(2, 2) = 9;
    matrix<double> m2 = m1 * m1;
    matrix<double> m3 = m2.sqrt();
    EXPECT_EQ(m3(0, 0), 1);
    EXPECT_EQ(m3(0, 1), 2);
    EXPECT_EQ(m3(0, 2), 3);
    EXPECT_EQ(m3(1, 0), 4);
    EXPECT_EQ(m3(1, 1), 5);
    EXPECT_EQ(m3(1, 2), 6);
    EXPECT_EQ(m3(2, 0), 7);
    EXPECT_EQ(m3(2, 1), 8);
    EXPECT_EQ(m3(2, 2), 9);
}

TEST(HyperMatrix, test4) {
    matrix<complex<double>> m1(10, 10);
    complex<double> c = complex<double>(-1, -1);

    for (size_t i = 0; i < 10; i++) {
        for (size_t j = 0; j < 10; j++) {
            m1(i, j) = c * complex<double>(i, -j) + complex<double>(j,-i); 
            
        }
    }

    matrix<complex<double>> m2 = m1 * m1;

    //check the real part and imaginary part of m2 
    EXPECT_EQ(m2(0, 0).real(), 0);
    EXPECT_EQ(m2(0, 1).real(), 0);
    EXPECT_EQ(m2(0, 2).real(), 0);
    EXPECT_EQ(m2(1, 0).real(), -3.4028236692093846e+38);
    EXPECT_EQ(m2(1, 1).real(),3.4028236692093846e+38);
    EXPECT_EQ(m2(1, 2).real(),3.4028236692093846e+38);
     
    
}
//test 5    
TEST( HyperMatrix, test5) {
    matrix<double> m1(3, 3);
    m1(0, 0) = 1;
    m1(0, 1) = 2;
    m1(0, 2) = 3;
    m1(1, 0) = 4;
    m1(1, 1) = 5;
    m1(1, 2) = 6;
    m1(2, 0) = 7;
}


//vector matrix operations tests
TEST(HyperMatrix, test6) {
    vector<double> v1(3);
    v1[0] = 1;
    v1[1] = 2;
    v1[2] = 3;
    matrix<double> m1(3, 3);
    m1(0, 0) = 1;
    m1(0, 1) = 2;
    m1(0, 2) = 3;
    m1(1, 0) = 4;
    m1(1, 1) = 5;
    m1(1, 2) = 6;
    m1(2, 0) = 7;
    m1(2, 1) = 8;
    m1(2, 2) = 9;
    vector<double> v2 = v1 * m1 ;
    EXPECT_EQ(v2[0], 14);
    EXPECT_EQ(v2[1], 32);
    EXPECT_EQ(v2[2], 50);

}

//TEST 7
//jacobian test

TEST(HyperMatrix, test7) {
    matrix<double> m1(3, 3);
    m1(0, 0) = 1;
    m1(0, 1) = 2;
    m1(0, 2) = 3;
    m1(1, 0) = 4;
    m1(1, 1) = 5;
    m1(1, 2) = 6;
    m1(2, 0) = 7;
    m1(2, 1) = 8;
    m1(2, 2) = 9;
    matrix<double> m2 =  m1.adjoint();
    EXPECT_EQ(m2(0, 0), 1);
    EXPECT_EQ(m2(0, 1), 4);
    EXPECT_EQ(m2(0, 2), 7);
    EXPECT_EQ(m2(1, 0), 2);
    EXPECT_EQ(m2(1, 1), 5);
    EXPECT_EQ(m2(1, 2), 8);
    EXPECT_EQ(m2(2, 0), 3);
    EXPECT_EQ(m2(2, 1), 6);
    EXPECT_EQ(m2(2, 2), 9);
}
//test 8 jacobian test. 
//define matrix a and b and test jacobian of a * b 

TEST(HyperMatrix, test8) {
  // Jacobi method
  // a1 : matrix
  // d : diagonal
  // v : eigenvectors
  // nrot : number of rotations

  matrix<double> a1(3, 3);
  a1(0, 0) = 1;
  a1(0, 1) = 2;
  a1(0, 2) = 3;
  a1(1, 0) = 4;
  a1(1, 1) = 5;
  a1(1, 2) = 6;
  a1(2, 0) = 7;
  a1(2, 1) = 8;
  a1(2, 2) = 9;
  matrix<double> b1(3, 3);
  b1(0, 0) = 1;
  b1(0, 1) = 2;
  b1(0, 2) = 3;
  b1(1, 0) = 4;
  b1(1, 1) = 5;
  b1(1, 2) = 6;
  b1(2, 0) = 7;
  b1(2, 1) = 8;
  b1(2, 2) = 9;
  matrix<double> c1 = a1 * b1;

  matrix<double> jac(3, 3); 
 
  double* diagonal = c1.as_diagonal() ;
  //set std::vector with diagonal elements 
  std::vector<double> diagonal_v(diagonal, diagonal + 3); 

  matrix<double> v = c1.eigenvectors(); 

  size_t nrot = 3*3;
  jacobi( (const matrix<double>&)c1,(std::vector<double>& ) diagonal_v, (matrix<double>& ) jac,nrot); 
  EXPECT_EQ(jac(0, 0), 0.087234237522280475); 
  EXPECT_EQ(jac(0, 1), 0.99435640192059038); 
  EXPECT_EQ(jac(0, 2), 0.060378255717132664); 
  EXPECT_EQ(jac(1, 0),  0.71968794783432832); 
  EXPECT_EQ(jac(1, 1),  -0.1048130033086031); 
  EXPECT_EQ(jac(1, 2), 0.68634065308667491); 
  EXPECT_EQ(jac(2, 0),  0.68879564861134202); 
  EXPECT_EQ(jac(2, 1), -0.016418900601680601); 
  EXPECT_EQ(jac(2, 2), -0.72476960074020269);
  

}
//test 9 eigenvalues and eigenvectors test 
//define matrix a and test eigenvalues and eigenvectors of a 

TEST(HyperMatrix, test9) {
  matrix<double> a1(3, 3);
  a1(0, 0) = 1;
  a1(0, 1) = 2;
  a1(0, 2) = 3; 
  a1(1, 0) = 4;
  a1(1, 1) = 5;
  a1(1, 2) = 6;
  a1(2, 0) = 7;
  a1(2, 1) = 8;
  a1(2, 2) = 9;
  matrix<double> eigenvalues = a1.eigenvalues();
  matrix<double> eigenvectors = a1.eigenvectors();
  EXPECT_EQ(eigenvalues(0, 0), 1);
  EXPECT_EQ(eigenvalues(1, 0),1); 
  EXPECT_EQ(eigenvalues(2, 0), 1); 
  EXPECT_EQ(eigenvectors(0, 0),  1); 
  EXPECT_EQ(eigenvectors(0, 1), 1); 
  EXPECT_EQ(eigenvectors(0, 2),1); 
  EXPECT_EQ(eigenvectors(1, 0),  1); 
  EXPECT_EQ(eigenvectors(1, 1),  1);
  EXPECT_EQ(eigenvectors(1, 2),1); 
  EXPECT_EQ(eigenvectors(2, 0), 1); 
  EXPECT_EQ(eigenvectors(2, 1),1); 
  EXPECT_EQ(eigenvectors(2, 2), 1);
}

//test 10 identity test 
//define matrix a and test identity of a 
TEST(HyperMatrix, test10) {
  matrix<double> a1(3, 3);
  a1(0, 0) = 1;
  a1(0, 1) = 2;
  a1(0, 2) = 3;
  a1(1, 0) = 4;
  a1(1, 1) = 5;
  a1(1, 2) = 6;
  a1(2, 0) = 7;
  a1(2, 1) = 8; 
  a1(2, 2) = 9;
  matrix<double> identity = a1.identity(); 
  EXPECT_EQ(identity(0, 0), 1);
  EXPECT_EQ(identity(0, 1), 0);
  EXPECT_EQ(identity(0, 2), 0);
  EXPECT_EQ(identity(1, 0), 0);
  EXPECT_EQ(identity(1, 1), 1);
  EXPECT_EQ(identity(1, 2), 0);
  EXPECT_EQ(identity(2, 0), 0);
  EXPECT_EQ(identity(2, 1), 0);
  EXPECT_EQ(identity(2, 2), 1);
}

#endif

