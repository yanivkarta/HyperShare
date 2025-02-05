#ifndef __TEST2_H__
#define __TEST2_H__
#include "../include/matrix.h"
#include "../include/mmatrix.h"
#include "../include/autoencoder.h"

//google test:
#include <gtest/gtest.h>

//test matrix
//add unit tests for isolation matrix

using namespace std;
using namespace provallo;

//test 11..20

//TEST 11 build spinor,tensor and autoencoder
//
TEST( test11, test11) {
    
     //build spinor
     matrix<double> m(2, 2);
     m(0, 0) = 1;
     m(0, 1) = 2;
     m(1, 0) = 3;
     m(1, 1) = 4;

     //build tensor
     matrix<double> t(2, 2);
     t(0, 0) = 1;
     t(0, 1) = 2;
     t(1, 0) = 3;
     t(1, 1) = 4;

     spinor<double> s(m);
     tensor<double> r( t, t, t);

     double sn = s.squaredNorm();
     double rn = r.squaredNorm(); 

     EXPECT_EQ(sn, 30);
     EXPECT_EQ(rn, 27000); 


     spinor<double> ml =s.angular_momentum(sn)  * rn;

     EXPECT_EQ(ml(0,0), -1053695.2397321474);
     EXPECT_EQ(ml(0,1), 2176478.6011729008);
     EXPECT_EQ(ml(1,0), 2927333.7937368592);
     EXPECT_EQ(ml(1,1), 1335852.571488901); 
}

//autoencoder test. 
//build autoencoder with 10 input, 10 hidden and 10 output neurons

TEST( test12, test12) {
    auto_encoder<double, double> a(10, 10, 10);
    std::uniform_real_distribution<> distribution(0.0, 1.0); 
    std::default_random_engine generator;
    std::vector<double> input(10);
    std::vector<double> output(10);
    
    for (int i = 0; i < 10; i++){
        
        input[i] = distribution(generator);
        output[i] =input[i] > 0.5;
    } 

    a.train(input.data(), output.data(),10);
    
    //predict :

    a.predict(input.data(), 10, output.data(), 10);

    
    auto score = a.score(input.data(), output.data(), 10);
    EXPECT_EQ(score[0]<0.5, true);
    EXPECT_EQ(score[1]<0.5, true);
    EXPECT_EQ(score[2]<0.5, true);
    EXPECT_EQ(score[3]<0.5, true);
    EXPECT_EQ(score[4]<0.5, true);
    EXPECT_EQ(score[5] <0.5, true);
    EXPECT_EQ(score[6]<0.5, true);
    EXPECT_EQ(score[7]<0.5, true);
    EXPECT_EQ(score[8]<0.5, true);
    EXPECT_EQ(score[9]<0.5, true);


}

 
//TEST 13 softmax test
TEST( test13, test13) {
    
    
    //create a matrix of real data and a vector of labels :
    matrix<double> X(10, 10);
    std::vector<unsigned int> y(10);
    for (int i = 0; i < 10; i++){
        for (int j = 0; j < 10; j++){
    
            std::default_random_engine generator;

            std::uniform_real_distribution<> distribution(0.0, 1.0); 
            

            X(i, j) = distribution(generator) *i + j*distribution(generator) + 2*i*j; 

        }

        y[i] = i; 
    }
    
    size_t nclasses = std::unique(y.begin(), y.end()) - y.begin(); 
    size_t nsamples = X.rows(); 
    size_t nfeatures = X.cols();
    size_t ndimensions = 2; 

    softmax_classifier<double,unsigned int> s(nclasses, nfeatures, ndimensions);

    s.train(X, y); 

    //create a matrix of real data and a vector of labels : 
    matrix<double> Xx(10, 10);
    std::vector<unsigned int> yy(10);
    for (int i = 0; i < 10; i++){
        for (int j = 0; j < 10; j++){
    
            std::default_random_engine generator;

            std::uniform_real_distribution<> distribution(0.0, 1.0);    
            

            Xx(i, j) = distribution(generator) *i + j*distribution(generator) + 2*i*j;

        }

        yy[i] = i;
    }
    
    //test predictions :
    std::vector<unsigned int> predictions(Xx.rows());
    //copy yy to predictions:
    std::copy(yy.begin(), yy.end(), predictions.begin());

    s.test(Xx, predictions); 

    
    //get labels :
    
    //test accuracy: 
    size_t correct_predictions = 0;  
    real_t accuracy = 0.,precision=0.,recall=0.,f1_score=0.; 
    real_t true_positives=0.,true_negatives=0.,false_positives=0.,false_negatives=0.; 
    real_t anomaly_ratio = 0.0; 
    for(size_t i=0;i<yy.size();i++)
    {
        if(yy[i] == predictions[i])
        {
            correct_predictions++; 
            true_positives++;
        }
        else
        {
            if(yy[i] > predictions[i])
            {
                false_negatives++;
            }
            else
            {
                false_positives++;
            }   

            if(yy[i] < predictions[i])
            {
                true_negatives++;
            }
            else
            {
                true_positives++;
            } 
        }
    }
    auto const _epsilon = 1e-10;
    accuracy = correct_predictions/yy.size(); 
    precision = true_positives/(true_positives+false_positives+_epsilon);
    recall = true_positives/(true_positives+false_negatives+_epsilon);    
    
    f1_score = 2*precision*recall/(precision+recall);

    anomaly_ratio = false_positives/(false_positives+true_negatives+_epsilon);

    
    //print results:
    std::cout << "Accuracy: " << accuracy << std::endl;
    std::cout << "Precision: " << precision << std::endl;
    std::cout << "Recall: " << recall << std::endl;
    std::cout << "F1 Score: " << f1_score << std::endl;
    std::cout << "Anomaly Ratio: " << anomaly_ratio << std::endl;
    
    EXPECT_EQ(accuracy>0.5, true); 
} 

#endif