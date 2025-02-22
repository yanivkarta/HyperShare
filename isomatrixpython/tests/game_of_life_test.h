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

using namespace std;
using namespace provallo;
 

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
TEST_F(GameOfLifeTest, GameOfLifeTest) {
        

}
#endif
