#ifndef __TEST2_H__
#define __TEST2_H__
#include "../include/matrix.h"
#include "../include/mmatrix.h"
#include "../include/autoencoder.h"
#include "../include/sampling_helper.h"
#include "../include/info_helper.h"
#include "../include/lstm.h"
#include "../include/fast_matrix_forest.h"
#include "../include/bit_vector_attribute.h"

//google test:
#include <gtest/gtest.h>

//test matrix
//add unit tests for isolation matrix

using namespace std;
using namespace provallo;
    //define MT-CO1 (Cytochrome C oxidase subunit I) gene sequence: 
    //        1 ccatccggaa gtgtatattt taattttacc aggattcggt ataatttctc acgttattag
    //61 acaggaaaga aataaaaagg aaacttttgg gaccttagga ataatctatg caataatggc
    //121 aattggttta ttaggtttca ttgtatgagc acatcatata tttacagttg gaatagatgt
    //181 tgatacccgt gcttatttta cttcagctac aataattatt gctgtgccca caggcattaa
    //241 aatttttagt tgactagcta ctttacatgg aacccaatta aattattcac cctcaatgct
    //301 atgatcttta ggatttgttt ttttatttac agtaggaggc ctaactgggg tagttcttgc
    //361 taattcttca attgatattg ttttacatga tacgtattat gtagtagctc attttcatta
    //421 tgttctttcc atgggagcag tatttgctat tatagctggc tttgttcact gatttccttt
    //481 atttactggt gtttcattga acaataaatt tttaaaaatc cagtttactt taatattttt
    //541 gggtgttaat ataacctttt tccctcaaca ttttttagga ttaagcggta tacctcgccg
    //601 gtattctgat tatcctgatg cttacacaac ttgaaatatt atttcatcaa ttggatcctt
    //661 aatttcttta attagaatca ttttattatt atttattatt tgagaagctt ttatttcagc
    //721 tcgaaaaaga ctatctcccc taagaataac atcttcaatt gaatgacttc aagaaatgcc
    //781 tcctgctgaa cacagatatt ctgaacttcc tatattgtct aatttc :

    const oligomer_sequence MT_CO1 = {
        {dnaC, dnaC, dnaA, dnaT, dnaC, dnaC, dnaG, dnaG, dnaA, dnaA},//ccatccggaa
        {dnaG,dnaT,dnaG,dnaT,dnaA,dnaT,dnaA,dnaT,dnaA,dnaT,dnaT,dnaT}, //gtgtatattt
        {dnaT,dnaA,dnaA,dnaT,dnaT,dnaT,dnaT,dnaA,dnaC,dnaC}, //taattttacc
        {dnaA,dnaG,dnaG,dnaA,dnaT,dnaT,dnaC,dnaG,dnaG,dnaT}, //aggattcggt
        {dnaA,dnaT,dnaA,dnaA,dnaT,dnaT,dnaT,dnaC,dnaT,dnaC}, //ataatttctc
        {dnaA,dnaC,dnaG,dnaT,dnaT,dnaA,dnaT,dnaT,dnaA,dnaG} ,//acgttattag
        {dnaA,dnaC,dnaA,dnaG,dnaG,dnaA,dnaA,dnaA,dnaG,dnaA}, //acaggaaaga
        {dnaA,dnaA,dnaT,dnaA,dnaA,dnaA,dnaA,dnaA,dnaG,dnaG},//aataaaaagg
        {dnaA,dnaA,dnaA,dnaC,dnaT,dnaT,dnaT,dnaT,dnaG,dnaG},//aaacttttgg
        {dnaG,dnaA,dnaC,dnaC,dnaT,dnaT,dnaA,dnaG,dnaG,dnaA},// gaccttagga
        {dnaA,dnaT,dnaA,dnaA,dnaT,dnaC,dnaT,dnaA,dnaT,dnaG},// ataatctatg 
        {dnaC,dnaA,dnaA,dnaT,dnaA,dnaA,dnaT,dnaG,dnaG,dnaC},//caataatggc
        {dnaA,dnaA,dnaT,dnaT,dnaG,dnaG,dnaT,dnaT,dnaT,dnaA},//aattggttta 
        {dnaT,dnaT,dnaA,dnaG,dnaG,dnaT,dnaT,dnaT,dnaC,dnaA},//ttaggtttca
        {dnaT,dnaT,dnaG,dnaT,dnaA,dnaT,dnaG,dnaA,dnaG,dnaC},// ttgtatgagc
        {dnaA,dnaC,dnaA,dnaT,dnaC,dnaA,dnaT,dnaA,dnaT,dnaA},// acatcatata 
        {dnaT,dnaT,dnaT,dnaA,dnaC,dnaA,dnaG,dnaT,dnaT,dnaG},//tttacagttg 
        {dnaG,dnaA,dnaA,dnaT,dnaA,dnaG,dnaA,dnaT,dnaG,dnaT},//gaatagatgt
        
        {dnaT,dnaG,dnaA,dnaT,dnaA,dnaC,dnaC,dnaC,dnaG,dnaT},//tgatacccgt
        {dnaG,dnaC,dnaT,dnaT,dnaA,dnaT,dnaT,dnaT,dnaT,dnaA},// gcttatttta 
        {dnaC,dnaT,dnaT,dnaC,dnaA,dnaG,dnaC,dnaT,dnaA,dnaC},//cttcagctac
        {dnaA,dnaA,dnaT,dnaA,dnaA,dnaT,dnaT,dnaA,dnaT,dnaT},// aataattatt 
        {dnaG,dnaC,dnaT,dnaG,dnaT,dnaG,dnaC,dnaC,dnaC,dnaA},//gctgtgccca 
        {dnaC,dnaA,dnaG,dnaG,dnaC,dnaA,dnaT,dnaT,dnaA,dnaA},//caggcattaa
        {dnaA,dnaA,dnaT,dnaT,dnaT,dnaT,dnaT,dnaA,dnaG,dnaT},//aatttttagt 
        {dnaT,dnaG,dnaA,dnaC,dnaT,dnaA,dnaG,dnaC,dnaT,dnaA},// tgactagcta 
        {dnaC,dnaT,dnaT,dnaT,dnaA,dnaC,dnaA,dnaT,dnaG,dnaG},// ctttacatgg 
        {dnaA,dnaA,dnaC,dnaC,dnaC,dnaA,dnaA,dnaT,dnaT,dnaA},//aacccaatta 
        {dnaA,dnaA,dnaT,dnaT,dnaA,dnaT,dnaT,dnaC,dnaA,dnaC},// aattattcac 
        {dnaC,dnaC,dnaT,dnaC,dnaA,dnaA,dnaT,dnaG,dnaC,dnaT},//cctcaatgct
        {dnaA,dnaT,dnaG,dnaA,dnaT,dnaC,dnaT,dnaT,dnaT,dnaA},//atgatcttta
        {dnaG,dnaG,dnaA,dnaT,dnaT,dnaT,dnaG,dnaT,dnaT,dnaT},// ggatttgttt 
        {dnaT,dnaT,dnaT,dnaT,dnaA,dnaT,dnaT,dnaT,dnaA,dnaC},//ttttatttac 
        {dnaA,dnaG,dnaT,dnaA,dnaG,dnaG,dnaA,dnaG,dnaG,dnaC},// agtaggaggc 
        {dnaC,dnaT,dnaA,dnaA,dnaC,dnaT,dnaG,dnaG,dnaG,dnaG},//ctaactgggg 
        {dnaT,dnaA,dnaG,dnaT,dnaT,dnaC,dnaT,dnaT,dnaG,dnaC},//tagttcttgc
        {dnaT,dnaA,dnaA,dnaT,dnaT,dnaC,dnaT,dnaT,dnaC,dnaA},//taattcttca 
        {dnaA,dnaT,dnaT,dnaG,dnaA,dnaT,dnaA,dnaT,dnaT,dnaG},//attgatattg 
        {dnaT,dnaT,dnaT,dnaT,dnaA,dnaC,dnaA,dnaT,dnaG,dnaA},//ttttacatga 
        {dnaT,dnaA,dnaC,dnaG,dnaT,dnaA,dnaT,dnaT,dnaA,dnaT},// tacgtattat 
        
        {dnaG,dnaT,dnaA,dnaG,dnaT,dnaA,dnaG,dnaC,dnaT,dnaC},//gtagtagctc 
        {dnaA,dnaT,dnaT,dnaT,dnaT,dnaC,dnaA,dnaT,dnaT,dnaA}, //attttcatta
        {dnaT,dnaG,dnaT,dnaT,dnaC,dnaT,dnaT,dnaT,dnaC,dnaC},// tgttctttcc
        {dnaA,dnaT,dnaG,dnaG,dnaG,dnaA,dnaG,dnaC,dnaA,dnaG},// atgggagcag 
        {dnaT,dnaA,dnaT,dnaT,dnaT,dnaG,dnaC,dnaT,dnaA,dnaT},//tatttgctat 
        {dnaT,dnaA,dnaT,dnaA,dnaG,dnaC,dnaT,dnaG,dnaG,dnaC},//tatagctggc
        {dnaT,dnaT,dnaT,dnaG,dnaT,dnaT,dnaC,dnaA,dnaC,dnaT},// tttgttcact 
        {dnaG,dnaA,dnaT,dnaT,dnaT,dnaC,dnaC,dnaT,dnaT,dnaT},//gatttccttt

        {dnaA,dnaT,dnaT,dnaT,dnaA,dnaC,dnaT,dnaG,dnaG,dnaT},//atttactggt 
        {dnaG,dnaT,dnaT,dnaT,dnaC,dnaA,dnaT,dnaT,dnaG,dnaA},//gtttcattga 
        {dnaA,dnaC,dnaA,dnaA,dnaT,dnaA,dnaA,dnaA,dnaT,dnaT},//acaataaatt 
        {dnaT,dnaT,dnaT,dnaA,dnaA,dnaA,dnaA,dnaA,dnaT,dnaC},//tttaaaaatc 
        {dnaC,dnaA,dnaT,dnaT,dnaT,dnaT,dnaA,dnaC,dnaT,dnaT},//cagtttactt
        {dnaT,dnaA,dnaA,dnaT,dnaA,dnaT,dnaT,dnaT,dnaT,dnaT},// taatattttt


        
        {dnaG,dnaG,dnaG,dnaT,dnaG,dnaT,dnaT,dnaA,dnaA,dnaT},//gggtgttaat 
        {dnaA,dnaT,dnaA,dnaA,dnaC,dnaC,dnaT,dnaT,dnaT,dnaT},//ataacctttt 
        {dnaT,dnaC,dnaC,dnaC,dnaT,dnaC,dnaA,dnaA,dnaC,dnaA},//tccctcaaca 
        {dnaT,dnaT,dnaT,dnaT,dnaT,dnaT,dnaA,dnaG,dnaG,dnaA},//ttttttagga 
        {dnaT,dnaT,dnaA,dnaA,dnaG,dnaC,dnaG,dnaG,dnaT,dnaA},//ttaagcggta 
        {dnaT,dnaA,dnaC,dnaC,dnaT,dnaC,dnaG,dnaC,dnaC,dnaG}, //tacctcgccg

        {dnaG,dnaT,dnaA,dnaT,dnaT,dnaC,dnaT,dnaG,dnaA,dnaT},//gtattctgat 
        {dnaT,dnaA,dnaT,dnaC,dnaC,dnaT,dnaG,dnaA,dnaT,dnaG},//tatcctgatg 
        {dnaC,dnaT,dnaT,dnaA,dnaC,dnaA,dnaC,dnaA,dnaA,dnaC},//cttacacaac 
        {dnaT,dnaT,dnaG,dnaA,dnaA,dnaA,dnaT,dnaA,dnaT,dnaT},//ttgaaatatt
        {dnaA,dnaT,dnaT,dnaT,dnaC,dnaA,dnaT,dnaC,dnaA,dnaA},// atttcatcaa 
        {dnaT,dnaT,dnaG,dnaG,dnaA,dnaT,dnaC,dnaC,dnaT,dnaT},//ttggatcctt
        
        {dnaA,dnaA,dnaT,dnaT,dnaT,dnaC,dnaT,dnaT,dnaT,dnaA},//aatttcttta 
        {dnaA,dnaT,dnaT,dnaA,dnaG,dnaA,dnaA,dnaT,dnaC,dnaA},//attagaatca 
        {dnaT,dnaT,dnaT,dnaT,dnaA,dnaT,dnaT,dnaA,dnaT,dnaT},//ttttattatt 
        {dnaA,dnaT,dnaT,dnaT,dnaA,dnaT,dnaT,dnaA,dnaT,dnaT},//atttattatt 
        {dnaT,dnaG,dnaA,dnaG,dnaA,dnaA,dnaG,dnaC,dnaT,dnaT},//tgagaagctt 
        {dnaT,dnaT,dnaA,dnaT,dnaT,dnaT,dnaC,dnaA,dnaG,dnaC},//ttatttcagc
        {dnaT,dnaC,dnaG,dnaA,dnaA,dnaA,dnaA,dnaA,dnaG,dnaA},//tcgaaaaaga 
        {dnaC,dnaT,dnaA,dnaT,dnaC,dnaT,dnaC,dnaC,dnaC,dnaC},//ctatctcccc 
        {dnaT,dnaA,dnaA,dnaG,dnaA,dnaA,dnaT,dnaA,dnaA,dnaC},//taagaataac
        {dnaA,dnaT,dnaC,dnaT,dnaT,dnaC,dnaC,dnaA,dnaA,dnaT},// atcttcaatt 
        {dnaG,dnaA,dnaA,dnaT,dnaG,dnaA,dnaC,dnaT,dnaT,dnaC},//gaatgacttc 
        {dnaA,dnaA,dnaG,dnaA,dnaA,dnaA,dnaT,dnaG,dnaC,dnaC},//aagaaatgcc
        {dnaT,dnaC,dnaC,dnaT,dnaG,dnaC,dnaT,dnaG,dnaA,dnaA},//tcctgctgaa 
        {dnaC,dnaA,dnaC,dnaA,dnaG,dnaA,dnaT,dnaA,dnaT,dnaT},//cacagatatt 
        {dnaC,dnaT,dnaG,dnaA,dnaA,dnaC,dnaT,dnaT,dnaC,dnaC},//ctgaacttcc 
        {dnaT,dnaA,dnaT,dnaA,dnaT,dnaT,dnaG,dnaT,dnaC,dnaT},//tatattgtct 
        {dnaA,dnaA,dnaT,dnaT,dnaT,dnaT,dnaC,dnaN,dnaN,dnaN} // aatttc :

    };

//test 11..20

//TEST 11 build spinor,tensor and autoencoder
//
TEST( spinor, test1) {
    
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

TEST( autoencoder, test12) {
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
TEST( softmax, test13) {
    
    
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

//TEST 11 - sampling_helper and fft / ftti transformation on random data samples
//
TEST(sampling_helper,test14) {
    
    //test fft capabilities of the sampling helper
    sampling_helper<double> sampler;
    
    matrix<double>  random_sample(10,10);
    std::default_random_engine generator;

    std::uniform_real_distribution<> distribution(0.0, 1.0);    
    

    for(size_t i=0;i<10;i++)
    {
        for (size_t j=0;j<10;j++)
        {
            random_sample(i,j) = distribution(generator) / (1+i + j*distribution(generator) + 2*i*j);
        }
    }
    matrix<complex<double>> result  = sampler.fft(random_sample);

    matrix<complex<double>> result2 = sampler.ifft(result); 

    matrix<complex<double>> delta = result2-result;

    EXPECT_EQ(delta.sum(), 0.0);

}

//bit vector tests: 
TEST(bit_vector,test1) {
    uint8_t aa = 1;
    uint8_t bb = 2;
    uint8_t cc = 3;

    bit_type<uint8_t,8> a(aa); 
    bit_type<uint8_t,8> b(bb);

    bit_type<uint8_t,8> c = a | b; 

    EXPECT_EQ(c[0], 1);
    EXPECT_EQ(c[1], 1);


    bit_type<uint8_t,8> d = a & b; 

    EXPECT_EQ(d[0], 0);
    EXPECT_EQ(d[1], 0);

    bit_type<uint8_t,8> e = a ^ b; 

    EXPECT_EQ(e[0], 1);
    EXPECT_EQ(e[1], 1);
    
    //test matrix of bits 
    matrix<bit_type<uint8_t,8>> m(2,2);
    m(0,0) = a; 
    m(0,1) = b; 
    m(1,0) = c; 
    m(1,1) = d; 

    EXPECT_EQ(m(0,0)[0], true);
    EXPECT_EQ(m(0,1)[1], true);
    EXPECT_EQ(m(1,0)[0], true);
    EXPECT_EQ(m(1,1)[1], false);

    //test bit vector of matrices 

    matrix<bit_type<uint8_t,8>> m2(2,2);
    m2 = m;
    m2(0,0) = a;

    EXPECT_EQ(m2(0,0)[0], true);
    EXPECT_EQ(m2(0,1)[1], true);
    EXPECT_EQ(m2(1,0)[0], true);
    EXPECT_EQ(m2(1,1)[1], false);

    //test matrix operations on bit vectors: 

    matrix<bit_type<uint8_t,8>> m3(2,2);
    m3(0,0) = a; 
    m3(0,1) = b; 
    m3(1,0) = c; 
    m3(1,1) = d;

    matrix<bit_type<uint8_t,8>> m4(2,2);
    m4= m3+m2;


    m4=m4-m3; 

    EXPECT_EQ(m4(0,0)[0], true);
    EXPECT_EQ(m4(0,1)[1], true);
    EXPECT_EQ(m4(1,0)[0], true);
    EXPECT_EQ(m4(1,1)[1], false);
}

//FAST Matrix tests: 
TEST(fast_matrix,test1) {
    

    matrix<double> m(10,10);
    //labels :
    std::vector<uint32_t> labels(10);
    std::default_random_engine generator;
    std::uniform_real_distribution<> distribution(0.0, 1.0);    
    for(size_t i=0;i<10;i++)
    {
        for (size_t j=0;j<10;j++)
        { 
            m(i,j) = distribution(generator) / (1+i + j*distribution(generator) + 2*i*j);
     
        }
        labels[i] = i;
    }

    super_tree<double,uint32_t> fmf(m,labels);

    
    matrix<double> m2(10,10);
    //labels :
    std::vector<uint32_t> labels2(10);
    std::default_random_engine generator2;
    std::uniform_real_distribution<> distribution2(0.0, 1.0);    
    for(size_t i=0;i<10;i++)
    {
        for (size_t j=0;j<10;j++)
        { 
            m2(i,j) = distribution2(generator2) / (1+i + j*distribution2(generator2) + 2*i*j);
     
        }
        labels2[i] = i;
    }

    fmf.fit(m2,labels2);


    std::vector<uint32_t> predictions(10);

    
    //regenerated data
    for(size_t i=0;i<10;i++)
    {
        for (size_t j=0;j<10;j++)
        { 
            m2(i,j) = distribution(generator) / (1+i + j*distribution(generator) + 2*i*j);
     
        }
        labels2[i] = i;

    }
    
    

    predictions = fmf.predict(m2,labels2);

    EXPECT_EQ(predictions[0], labels2[0]);
    EXPECT_EQ(predictions[1], labels2[1]);
    EXPECT_EQ(predictions[2], labels2[2]);
    EXPECT_EQ(predictions[3], labels2[3]);
    EXPECT_EQ(predictions[4], labels2[4]);
    EXPECT_EQ(predictions[5], labels2[5]);
    EXPECT_EQ(predictions[6], labels2[6]);
    EXPECT_EQ(predictions[7], labels2[7]);
    EXPECT_EQ(predictions[8], labels2[8]);
    EXPECT_EQ(predictions[9], labels2[9]);
    
    //test accuracy: 
    size_t correct_predictions = 0;  
    real_t accuracy = 0.,precision=0.,recall=0.,f1_score=0.; 
    real_t true_positives=0.,true_negatives=0.,false_positives=0.,false_negatives=0.; 
    real_t anomaly_ratio = 0.0; 
    for(size_t i=0;i<10;i++)
    {
        if(labels2[i] == predictions[i])
        {
            correct_predictions++; 
            true_positives++;
        }
        else
        {
            if(labels2[i] > predictions[i])
            {
                false_negatives++;
            }
            else
            {
                false_positives++;
            }
        }
    }
    auto const _epsilon = 1e-10;
    accuracy = correct_predictions/10;
    precision = true_positives/(true_positives+false_positives+_epsilon);
    recall = true_positives/(true_positives+false_negatives+_epsilon);
    f1_score = 2*precision*recall/(precision+recall+_epsilon);
    anomaly_ratio = false_positives/(10+_epsilon);
    EXPECT_EQ(accuracy>0.99, true);
    EXPECT_EQ(precision>0.99, true);
    EXPECT_EQ(recall>0.99, true);
    EXPECT_EQ(f1_score>0.99, true);
    EXPECT_EQ(anomaly_ratio<0.01, true);

    //log the results: 
    std::cout<<"Accuracy: "<<accuracy<<std::endl;
    std::cout<<"Precision: "<<precision<<std::endl;
    std::cout<<"Recall: "<<recall<<std::endl;
    std::cout<<"F1 Score: "<<f1_score<<std::endl;
    std::cout<<"Anomaly Ratio: "<<anomaly_ratio<<std::endl;    
}
TEST( evaluation_metrics, test3)
{
    //loss,huber loss: 
    matrix<double> m(10,10);
    //labels :
    std::vector<uint32_t> labels(10),labels2(10);
    std::default_random_engine generator;
    std::uniform_real_distribution<> distribution(0.0, 1.0);    
    for(size_t i=0;i<10;i++)
    {
        for (size_t j=0;j<10;j++)
        { 
            m(i,j) = distribution(generator) / (1+i + j*distribution(generator) + 2*i*j);
     
        }
        labels[i] = i;
    } 

    matrix<double> m2(10,10);
    for(size_t i=0;i<10;i++)
    {
        for (size_t j=0;j<10;j++)
        { 
            m2(i,j) = distribution(generator) / (1+i + j*distribution(generator) + 2*i*j);
     
        }
        labels2[i] = i;
    }
 
    auto const _epsilon = 1e-10; 
    real_t loss=0.,huber_loss=0.,mae=0.,mse=0.,rmse=0.; 
    for (size_t i=0;i<10;i++)
    {
        for (size_t j=0;j<10;j++)
        {
            loss += std::abs(m(i,j)-m2(i,j)); 
            huber_loss += std::abs(m(i,j)-m2(i,j));
            mae += std::abs(m(i,j)-m2(i,j));
            mse += std::pow(m(i,j)-m2(i,j),2);
            rmse += std::pow(m(i,j)-m2(i,j),2);
        }
    }
    loss = loss/100;
    huber_loss = huber_loss/100;
    mae = mae/100;
    mse = mse/100;
    rmse = std::sqrt(rmse/100);
    EXPECT_EQ(loss<0.1, true);
    EXPECT_EQ(huber_loss<0.1, true);
    EXPECT_EQ(mae<0.1, true);
    EXPECT_EQ(mse<0.2, true);
    EXPECT_EQ(rmse<0.2, true);

}


bool validate_sequence(const oligomer_sequence& sequence)
{
    bool valid = true;
    for (size_t i = 0; i < sequence.size(); i++)
    {
        for (size_t j =0;j < sequence[i].size(); j++)
        {
            if (sequence[i][j] != dnaA && sequence[i][j] != dnaC && sequence[i][j] != dnaG && sequence[i][j] != dnaT) 
            {
                valid = false;
                break;
            }            
            //check if the sequence is valid, neucloeotide A,C,G,T sequence can be used 
            //only once in the sequence, repetition is not allowed unless it is the last one in the sequence 
            //if the sequence is not valid, return false
            if (i != sequence.size() - 1 && sequence[i][j] == sequence[i + 1][j])
            {
                valid = false;
                break;
            }
        }
    }
    return valid;   
}
bool is_valid_addition(const oligomer_sequence& sequence1, dna_base base , const size_t index1, const size_t index2) 
{
    bool valid = true;
    //check if the indexes are valid
    if (index1 >= sequence1.size() || (index2-1) >= sequence1[index1].size()) 
    {
        return false;
    }

    if(sequence1[index1][index2-1] == base ) 
    {
        valid = false;
    }
    return valid;

}
TEST(bio_simulation, test4)
{
    //initialize generators, use helmholz machine:
    gaussian_spike_train_generator<real_t> generator    ;
    //set inputs:
    


    std::vector<uint8_t> labels(10),predictions(10);
    

    //oligoneucleotide :
    oligomer_sequence o_sequence(10);   

    //generate data:
    for (size_t i=0;i<10;i++)
    {
        
        auto x = generator.generate(); 
        auto y = generator.generate();
        //convert to oligonucleotide sequence:
        for (size_t j=0;j<10;j++)
        {
            //if x==0 and y==0: A 
            //if x>0 and y==0: C
            //if x==0 and y>0: G
            //if x>0 and y>0: T
            bool valid = false;
            
            auto previous_tail = j>0?o_sequence[i][j-1]:dnaN;
            dna_base previous = previous_tail;
            //check if the sequence is valid, neucloeotide A,C,G,T sequence can be used 
            //only once in the sequence, repetition is not allowed unless it is the last one in the sequence 
            

            dna_base b = (x[j]==0. && y[j]==0.) ? dnaA : (x[j]>0. && y[j]==0.) ? dnaC : (x[j]==0.&& y[ j]>0.) ? dnaG : dnaT; 
            if(previous_tail == b)
            {
                if (dnaA == b)
                {
                    x = generator.generate();
                    y = generator.generate();
                }
                
                //not valid,must change b to a different base 
                b ^=   previous_tail;
                if (b == previous_tail)
                {
                    //AA

                    b^=b.flip();
                }
                valid = true;
            }
            
            

            //check if the sequence is valid, neucloeotide A,C,G,T sequence can be used 
            //only once in the sequence, repetition is not allowed unless it is the last one in the sequence 
            //if the sequence is not valid, return false
            
            o_sequence[i].push_back(b);
            

            
        }
    } //    

    std::cout<<"bit sequence: "<<std::endl;
    std::cout<<o_sequence<<std::endl; 
    std::cout<<"dna sequence: "<<std::endl;
    for (size_t i=0;i<10;i++)
    {
        for (size_t j=0;j<10;j++)
        {
            char x = (o_sequence[i][j] == dnaA) ? 'A' : (o_sequence[i][j] == dnaC) ? 'C' : (o_sequence[i][j] == dnaG) ? 'G' : (o_sequence[i][j] == dnaT ) ?'T' : 'X'; 
            std::cout<<x;
        }
        //std::cout<<std::endl;
    }
    std::cout<<std::endl;
    //step forward 10 times:
    size_t steps = 10;
    auto old_sequence = o_sequence;

    do 
    {
    
    o_sequence.clear();
    o_sequence.resize(10);
    // regenerate data:
    for (size_t i=0;i<10;i++)
    {
        auto x = generator.generate(); 
        auto y = generator.generate();
        //convert to oligonucleotide sequence:
        for (size_t j=0;j<10;j++)
        {
            //if x==0 and y==0: A 
            //if x>0 and y==0: C
            //if x==0 and y>0: G 
            //if x>0 and y>0: T
            dna_base b = (x[j]==0. && y[j]==0.) ? dnaA : (x[j]>0. && y[j]==0.) ? dnaC : (x[j]==0.&& y[ j]>0.) ? dnaG : dnaT; 
            o_sequence[i].push_back(b);
        }    
    } //    
    
    std::cout<<"dna sequence: "<<std::endl;
    for (size_t i=0;i<10;i++)
    {
        for (size_t j=0;j<10;j++)
        {
            char x = (o_sequence[i][j] == dnaA) ? 'A' : (o_sequence[i][j] == dnaC) ? 'C' : (o_sequence[i][j] == dnaG) ? 'G' : (o_sequence[i][j] == dnaT ) ?'T' : 'X'; 
            std::cout<<x;
        }
        //std::cout<<std::endl;
    }
    std::cout<<std::endl;   
    }
    while (steps--);


    //xor o_sequence with old_sequence:
    for (size_t i=0;i<10;i++)
    {
        for (size_t j=0;j<10;j++)
        {
            o_sequence[i][j] ^= old_sequence[i][j];
        }
    }
    //print o_sequence:

    std::cout<<"xor dna  subsequence: "<<std::endl;
    for (size_t i=0;i<10;i++)
    {
        for (size_t j=0;j<10;j++)
        {   
            char x = (o_sequence[i][j] == dnaA) ? 'A' : (o_sequence[i][j] == dnaC) ? 'C' : (o_sequence[i][j] == dnaG) ? 'G' : (o_sequence[i][j] == dnaT ) ?'T' : 'X'; 
            std::cout<<x;
        }    
        std::cout<<std::endl;
    }

        

    
}

TEST(bio_simulation, MT_C01)
{
    // test MT_C01 over generated sequences 

    oligomer_sequence seq ;
    boltzman_base<real_t> bolz(MT_CO1.size(),MT_CO1[0].size(),MT_CO1.size(),MT_CO1[0].size(),1);        
    vector<vector<real_t> > MT_CO1_real(MT_CO1.size());
    for(size_t i=0;i<MT_CO1.size();++i)
    {
        MT_CO1_real[i].resize( MT_CO1[i].size());

        for(size_t j=0;j<MT_CO1[i].size();++j)
        {
            //transform dna values to real values 
            auto value = MT_CO1[i][j];

            MT_CO1_real[i][j]= (value==dnaA) ? 0.0 : (value==dnaC) ? -1.0 : (value==dnaG) ? 1.0 : (value==dnaT) ? 2.0 : 0.0; 


            //(value==0.0) ? dnaA : (value<0.0) ? dnaC : (value>0.0) ? dnaG : (value!=value) ? dnaT : dnaN; 
        }
        
        bolz.refine(MT_CO1_real[i]);

        
    }
    //process input for bolzman: 

    cout<<"Boltz DNA:"<<endl;
    //fit the generator : 
    seq.resize(MT_CO1.size());
    for (size_t i=0;i<MT_CO1.size();i++) 
    {
        real_t step = bolz.step();
        const auto& generated_real = bolz.generate();
        //translate generated_real to oligo
        //validate GENERATED_REAL > 1 
        //
        EXPECT_EQ(generated_real.size()>1,true);
        
        size_t j=0;
        seq[i].resize(generated_real.size());
        for(auto & value : generated_real)
        {
            auto& prev = seq[i][j];
            seq[i][j++] = (value==0.0) ? dnaA : (value<0.0) ? dnaC : (value>0.0) ? dnaG : (value!=value) ? dnaT : dnaN; 
            if(prev==seq[i][j-1])
            {
                
                prev++;
                if(seq[i][j]==seq[i][j-1])
                {
                    seq[i][j-1]++;
                }
            }
            else 
            {
                prev--;
            }
            string x = (seq[i][j-1]==dnaA)?"A":(seq[i][j-1]==dnaC)?"C":(seq[i][j-1]==dnaG)?"G":(seq[i][j-1]==dnaT)?"T":"X";
            cout<<x;
        }
        
        
        
    }
    //compare to   Cytochrome C oxidase subunit I:
    //check distance from generated oligo sequence to MT_C01:
    real_t distance = 0.0;
    for (size_t i=0;i<MT_CO1.size();i++) 
    {
        for (size_t j=0;j<MT_CO1[i].size();j++)
        {
            distance += (seq[i][j] != MT_CO1[i][j]) ? 1.0 : 0.0;
        }
    }
    distance /= (MT_CO1.size() * MT_CO1[0].size());
    cout<<endl; 
    cout<<"distance: "<<distance<<endl;
    cout<<endl; 
    EXPECT_NEAR(distance,0.0,1.0);

    //print oligo sequence:
    cout<<"Oligo sequence: "<<endl;
    for (size_t i=0;i<MT_CO1.size();i++) 
    {
        for (size_t j=0;j<MT_CO1[i].size();j++)
        { 
            string x = (seq[i][j]==dnaA)?"A":(seq[i][j]==dnaC)?"C":(seq[i][j]==dnaG)?"G":(seq[i][j]==dnaT)?"T":"X";
            cout<<x;
        }
     
    }
    cout<<endl; 



}
#endif