#ifndef __GAME_OF_LIFE_TEST_H__
#define __GAME_OF_LIFE_TEST_H__

//google test:
#include <gtest/gtest.h>


#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <thread>
#include <mutex>
#include <memory>
#include <functional>
#include <utility>
#include <tuple>
#include <map>

#include "../include/matrix.h"

#include "../include/sampling_helper.h" //for sampling helpers
#include "../include/info_helper.h" //for info helpers
//bit vector
#include "../include/bit_vector_attribute.h"
#include "../include/optimizers.h"

using namespace std;
using namespace provallo;
using namespace provallo::GENETIC;

class GameOfLifeTest : public ::testing::Test {
    //conway game of life
    int width = 256;
    int height = 256;
    matrix<int> board;

    int num_iterations = 100;
    
public:
    GameOfLifeTest():board(width, height) {

        board.randomize();
        for (int i = 0; i < num_iterations; i++) {
          //purple text :
          printf("\033[35m");
          printf ("Iteration %d\n", i);
          std::string s;
          for ( size_t x = 0; x < width; x++ ) {
              s+="=";
          }
          //end purple text
          printf("%s\n\033[0m", s.c_str());

          //step through the game of life
          for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
              int neighbors = 0;
              for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                  if (dx == 0 && dy == 0) {
                    continue;
                  }
                  int nx = x + dx;
                  int ny = y + dy;
                  if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                    continue;
                  }
                  if (board(nx, ny) == 1) {
                    neighbors++;
                  }
                }
              }
              if (board(x, y) == 1 && neighbors < 2) {
                board(x, y) = 0;
              } else if (board(x, y) == 1 && neighbors > 3) {
                board(x, y) = 0;
              } else if (board(x, y) == 0 && neighbors == 3) {
                board(x, y) = 1;
              }
            }
          }

          //print the board
          for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
              if (board(x, y) >= 1) {
                if (board(x, y) == 1) {
                  //blink the board
                  printf("\x1B[1;11m*\x1B[0m");
                }
                else {
                  printf("\x1B[1;22m*\x1B[0m");
                }
                
              } else {
                //blink the board
                printf("\x1B[1;31m.\x1B[0m");
                
              }
            }
            printf("\n");
          }
          printf("\n"); 

        }

        //print the board matrix:
        cout << board << endl;

    }
        

    
    virtual ~GameOfLifeTest() {
    }
};


class genetic_algorithm_test : public ::testing::Test {
    
    //not a GA but a simulation of a GA using bit vectors 
    //for testing.

    //define chromosomes with bit vectors

    //    bit_type<uint64_t, 64> chromosome;
    
    //population

    //mutation rate
    double mutation_rate = 0.01;
    std::vector<bit_type<uint64_t, 64>> population;

    //fitness function
    double fitness_function(bit_type<uint64_t, 64> chromo) {
        //fitness function
        double fitness = 0.0;
        //compute fitness based on distance to goal 
        for (auto& chromosome : population) {
            double distance = 0.0;
            for (int i = 0; i < 64; i++) {
                if (chromosome!=chromo) {
                    distance += abs(chromosome[i] - chromo[i]); 
                }
            }
            fitness += distance;
        }
        fitness /= population.size();

        return fitness;
    }
    //crossover function
    bit_type<uint64_t, 64> crossover_function(bit_type<uint64_t, 64> parent1, bit_type<uint64_t, 64> parent2) {
        //crossover function
        bit_type<uint64_t, 64> child = parent1|parent2 ;
        return child;

    }
    //mutation function
    bit_type<uint64_t, 64> mutation_function(bit_type<uint64_t, 64> chromosome) {
        //mutation function
        //generate a random bit to flip 

        //c++ random, uniform int distribution (0-63):
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(0, 63);

        int bit_to_flip = dis(gen);
        //flip the bit
        
        bit_type<uint64_t, 64> mutated_chromosome = chromosome;
        mutated_chromosome.flip(bit_to_flip);


        return mutated_chromosome;
    }
    //selection function
    bit_type<uint64_t, 64> selection_function(std::vector<bit_type<uint64_t, 64>> population) {
        //selection function
        bit_type<uint64_t, 64> best_chromosome = population[0];
        for (auto &x : population)
        {
            if (fitness_function(x) > fitness_function(best_chromosome)) {
                best_chromosome = x;
            }
        }
        return best_chromosome;
    }

    //step function
    bit_type<uint64_t, 64> step_function(bit_type<uint64_t, 64> chromosome) {
        //step function
        //crossover
        random_device rd;
        mt19937 gen(rd());

        bit_type<uint64_t, 64> parent1 = selection_function(population);
        bit_type<uint64_t, 64> parent2 = selection_function(population);
        bit_type<uint64_t, 64> child = crossover_function(parent1, parent2);
        //mutation
        std::normal_distribution<double> distribution(0, 1); 
        double mutation_prob = distribution(gen);
        if (mutation_prob < mutation_rate) {
            child = mutation_function(child);
        }

        return child;
    }
    //solve function
    bit_type<uint64_t, 64> solve_function(bit_type<uint64_t, 64> chromosome) {
        //solve function
        //step function
        bit_type<uint64_t, 64> next_chromosome = step_function(chromosome);
        return next_chromosome;
    }   

    


    public:

    //initialize population:

    genetic_algorithm_test(): population(1000) {
        std::random_device rd;
        std::mt19937 gen(rd());
        //standard normal distribution
        std::uniform_int_distribution<uint64_t> distribution(0, 1); 

        for (auto &x : population) {
            //init random population
            x = bit_type<uint64_t, 64>(distribution(gen)); 
        }

        //solve the problem,calculate the number of steps it takes to solve the problem 
        bit_type<uint64_t, 64> chromosome = solve_function(population[0]); 
        int steps = 0;
        int generation = 0;
        while (chromosome != population[population.size()-1]) {

            for (auto &x : population) {
                //init random population
                x = bit_type<uint64_t, 64>(distribution(gen)); 
                chromosome = solve_function(x);
                if (chromosome == population[population.size()-1]) { 
                    break;
                }
                steps++;    
            }
            //step:
            generation++;
            std::cout<<"generation: "<<generation<<endl;
            
        }
        cout << "steps: " << steps << "//" << population.size() <<   endl; 
        cout << "chromosome: " << chromosome << endl;



    }
    virtual ~genetic_algorithm_test() {
    }   


};

TEST_F(GameOfLifeTest, GameOfLifeTest) {

}

TEST_F(genetic_algorithm_test, GeneticAlgorithmTest) {
    
}
TEST(multigrid, Multigrid) {

    //multigrid algorithm test with matrix 
    matrix<double> m(10, 10);
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            m(i, j) = i + j;
        }
    }
    matrix<double> m2(10, 10);
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            m2(i, j) = i + j;
        }
    }
    
    slvsml(m, m2);
    EXPECT_GE(m(0,0),0.0);
    EXPECT_GE(m(0,1),0.0);
    EXPECT_GE(m(1,0),0.0);
    EXPECT_LE(m(1,1),0.0);
    //full multigrid algorithm ( no recursion) :
    mg((m.size1()+1)/2, m, m2);

    //print the result:
    for (int i = 0; i < 10; i++) {

        for (int j = 0; j < 10; j++) {
            //if boundary condition, print '.'
            if (i == 0 || i == 9 || j == 0 || j == 9) {
                std::cout << "=" << "    ";
            }
            else {
            std::cout << m(i, j) << " ";
            }
        }
        std::cout << std::endl;
    }

    
}
template <typename T>
vector<T> quantum_convex(vector<T> demand, vector<T> wind, vector<T> solar) {

    //quantum convex optimization, create quantum oracle
    //quantum oracle:
    matrix<matrix<complex<T>>> quantum_oracle(3, 3);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            quantum_oracle(i, j).resize(demand.size(), demand.size()); 
            for (int k = 0; k < demand.size() && k < wind.size() && k < solar.size() ; k++) {
                for (int l = 0; l < demand.size() && l < wind.size() && l < solar.size() ; l++) {
                    
                    quantum_oracle(i, j)(k, l) = demand[k] + wind[k] + solar[k];


                }
            }
        }
    }
    matrix<complex<T>>& prev = quantum_oracle(0, 0);
    for (auto &x : quantum_oracle) {
        
        //use quasi-Newton method to update the quantum oracle 
        for (size_t i=0; i < x.size1(); i++) {
            for (size_t j=0; j < x.size2(); j++) {
                x(i, j) = prev(i, j) * std::complex<T>(sqrt(2), 0) - x(i, j) * std::complex<T>(sqrt(2), 0); 
                //apply the gradient descent method: 
                x(i, j) += 0.1 * (prev(i, j) - x(i, j)); 

            }
        }


    }
    //calculate the convex from the quantum oracle, and return the convex function 

    matrix<T> convex(demand.size(), demand.size());
    for (int i = 0; i < demand.size(); i++) {
        for (int j = 0; j < demand.size(); j++) {
            convex(i, j) = 0;
            for (int k = 0; k < 3; k++) {
                for (int l = 0; l < 3; l++) {
                    convex(i, j) += quantum_oracle(k, l)(i, j).real(); 
                }
            }
        }
    }
    //return convex as a vector
    vector<T> convex_vector;
    for (int i = 0; i < convex.size1(); i++) {
        for (int j = 0; j < convex.size2(); j++) {
            convex_vector.push_back(convex(i, j));
        }
    }
    return convex_vector;
    
}
TEST(QuantumConvex, QuantumConvex) {

    vector<double> demand = { .1, .2, .3, .4, .5, .6, .7, .8, .9, 1 }; 
    vector<double> wind = { .1, .2, .3, .4, .5, .6, .7, .8, .9, 1 };
    vector<double> solar = { .1, .2, .3, .4, .5, .6, .7, .8, .9, 1 };

    //demand range [0, 1], wind range [0, 1], solar range [0, 1] 
    //solar = wind + demand
    //wind = solar - demand
    //demand = solar - wind

    for (int i = 0; i < demand.size()&& i < wind.size()&& i < solar.size(); i++) { 
        solar[i] = wind[i] + demand[i];
        wind[i] = solar[i] - demand[i];
        demand[i] = solar[i] - wind[i];        
    }
    vector<double> convex = quantum_convex(demand, wind, solar);
    //draw the convex function over a plane of demand and wind 
    //use 3d plot to draw the convex function over a plane of demand and wind 
    //open gnuplot file :
    std::ofstream file;
    file.open("convex.txt"); 
    for (int i = 0; i < convex.size(); i++) {
        file << convex[i] << std::endl;
    }
    file.close();
    file.open("demand.txt"); 
    for (int i = 0; i < demand.size(); i++) {
        file << demand[i] << std::endl;
    }
    file.close();
    file.open("wind.txt"); 
    for (int i = 0; i < wind.size(); i++) {
        file << wind[i] << std::endl;
    }
    file.close();
    
    file.open("quantum_convex.gnuplot"); 
    file << "set terminal png" << std::endl;
    file << "set output 'quantum_convex.png'" << std::endl;
    file << "set xlabel 'demand'" << std::endl;
    file << "set ylabel 'wind'" << std::endl;
    file << "set zlabel 'solar'" << std::endl;
    file << "set title 'quantum convex'" << std::endl;
    file << "set pm3d" << std::endl;
    file << "set palette rgbformulae 8, 10, 12" << std::endl;
//  set range for imaginary part and real part 
    file << "set xrange [-1:1]" << std::endl;
    file << "set yrange [-1:1]" << std::endl;
    file << "set zrange [-1:1]" << std::endl;
    //allow imaginary part to be negative 
    file << "set cbtics -1, 0.25, 1" << std::endl; 
    //set range for real part
    file << "set cbrange [-1:1]" << std::endl;
    //set range for imaginary part
    file << "set cbrange [-1:1]" << std::endl; 

    //draw demand as surface

    file << "unset surface" << std::endl;

    //draw solar as surface 
    file << "set view 60, 60" << std::endl;
    file << "set hidden3d" << std::endl;
    file << "unset surface" << std::endl;
    //create a function to draw the convex function 
    file<<"f(x,y) =  x*cos(y) + y*sin(x) - sqrt(x*x + y*y) "<<std::endl;

    //isoplot f(x,y) with pm3d notitle

    file << "splot f(x,y) with pm3d notitle" << std::endl;

    file << "set view 0, 90" << std::endl;

    file << "splot 'convex.txt' using f(1 - sqrt(1)*sin(1),sqrt(1)*cos(1) - sqrt(1)*sin(1) ) with pm3d notitle" << std::endl;

    file << "unset surface" << std::endl;
    file << "set view 60, 90" << std::endl;

    file << "splot 'demand.txt' using f(1,cos(1)) with pm3d notitle" << std::endl; 

    file << "unset surface" << std::endl;
    file << "set view 90, 0" << std::endl;

    file << "splot 'wind.txt' using f(1,-sin(1)) with pm3d notitle" << std::endl;
    
    //plot solar as vector arrows   
    file << "unset surface" << std::endl;
    file << "set view 0, 90" << std::endl;
    file << "splot 'solar.txt' using f(1,-sin(1)) with vectors notitle" << std::endl;
    file << "unset surface" << std::endl;
    
    

    file.close();

    

}
#endif
