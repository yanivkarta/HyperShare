
**ISOLATION MATRIX (Super tree) Decoupled from provallo's meta-learning engine. 
This package contains a wrapper for python and a small test script. 



To use:
1. Create a build directory (e.g. mkdir build;cd build; cmake ..) 
2. build the .so (e.g. make -j8;make install)
3. copy the test.py to the build directory and run python3 ./test.py
4. output should be : 
```
Signature of the fit function:  {} None fit FastMatrixForest.fit
Signature of the predict function:  {}
Fitting the forest
Forest fitted successfully
Forest fitted successfully
Time taken to fit the forest:  5.4836273193359375e-06
Anomaly scores:  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
Length of the anomaly scores:  1000
Time taken to predict the anomaly scores:  0.006739377975463867
Time taken to delete the forest:  1.1920928955078125e-05

----------------------------------------------------------------------
Ran 0 tests in 0.000s

OK
```





A few example of how to use the isolation matrix (super tree) : 
1) comparison to isolation forests on anomaly datasets :

``` 

// datasets = ["http", "smtp", "SA", "SF", "shuttle", "forestcover"]
// saves 
 void compare_forest_to_hyperplane_matrix() 
 {
    auto datasets = std::vector<std::string>{"http", "smtp", "SA", "SF", "shuttle", "forestcover"}; 
    
    //real_t outlier_ratio = 0.1;  
    //count outlier values into ratio once predict is evaluated. 
     
    
    std::vector<real_t> fpr_vector; 
    std::vector<real_t> tpr_vector; 
    std::vector<real_t> roc_area_vector; 
    
    std::vector<real_t> precision_recall_area;
    //read the datasets and create super-tree 
    std::vector<provallo::hashed_bag_of_words> vectorizers; 
    //real_t outlier_ratio = 0.1; 
    
    //count outlier values into ratio once predict is evaluated.
    //read datasets into matrices of X,y from numpy(.npy) formatted files :
    //create a super-tree from the datasets 
    //load the files:
    std::cout<<"[+] loading datasets : "<<std::endl; 
    for(auto& dat : datasets)
    {
      //read X,y
      std::string X_file = std::string("./db/anomaly/")+dat+std::string("_X.csv"); 
      std::string y_file = std::string("./db/anomaly/")+dat+std::string("_y.csv"); 
      std::cout<<"[+] loading X : "<<X_file<<std::endl; 
      std::cout<<"[+] loading y : "<<y_file<<std::endl; 
      //load X,y 
      std::vector<std::vector<real_t>> X_raw; 
      provallo::matrix<real_t> X;  
      provallo::matrix<real_t> y; 
      
      std::ifstream X_stream(X_file); 
      std::ifstream y_stream(y_file);
      std::string line;
      while(std::getline(X_stream,line))
      {
        std::vector<real_t> row; 
        std::stringstream ss(line); 
        std::string value; 
        if(line.length() <2)
        {
          continue; 
        }
        while(std::getline(ss,value,','))
        {
          row.push_back(std::stod(value)); 
        }
        X_raw.push_back(row); 
      }
      X_stream.close();
      //convert X_raw to X matrix

      if(X_raw.size() == 0 || X_raw[0].size() == 0)
      {
        if(X_raw.size()>0)
        {
          //maybe ignore the first row 
          X_raw.erase(X_raw.begin());
          if(X_raw.size() > 0)
            std::cout<<"[+] error loading X : "<<X_file<<",size = "<<std::to_string(X_raw.size())<<std::to_string(X_raw[0].size())<< std::endl; 
          continue;
          
        }
        std::cout<<"[-] error loading X : "<<X_file<<std::endl; 
        continue; 
      } 
      X.resize(X_raw.size(),X_raw[0].size());
      for(size_t i=0;i<X_raw.size();i++)
      {
        for(size_t j=0;j<X_raw[0].size();j++)
        {
          X(i,j) = X_raw[i][j]; 
          //check value is not nan or inf
          if(std::isnan(X(i,j)) || std::isinf(X(i,j)))
          {
            //replace -inf with -1.0,inf with 1.0,nan with 0.0 
            if(std::isinf(X(i,j)))
            {
              if(X(i,j) < 0)
              {
                X(i,j) = -1.0; 
              }
              else
              {
                X(i,j) = 1.0; 
              }
            }
            else
            {
              X(i,j) = 0.0; 
            } 

          } 
        }
      }
      //load y
      std::vector<std::vector<real_t>> y_raw;
      while(std::getline(y_stream,line))
      {
        std::vector<real_t> row; 
        std::stringstream ss(line); 
        std::string value; 
        while(std::getline(ss,value,','))
        {
          row.push_back(std::stod(value)); 
        }
        y_raw.push_back(row); 
      }
      y_stream.close();
      //convert y_raw to y matrix
      y.resize(y_raw.size(),y_raw[0].size());
      for(size_t i=0;i<y_raw.size();i++)
      {
        for(size_t j=0;j<y_raw[0].size();j++)
        {
          y(i,j) = y_raw[i][j]; 
          //check value is not nan or inf 
          if(std::isnan(y(i,j)) || std::isinf(y(i,j)))
          {
            //replace -inf with -1.0,inf with 1.0,nan with 0.0 
            if(std::isinf(y(i,j)))
            {
              if(y(i,j) < 0)
              {
                y(i,j) = -1.0; 
              }
              else
              {
                y(i,j) = 1.0; 
              }
            }
            else
            {
              y(i,j) = 0.0; 
            } 

          }

        }
      }
     
     //get number of classes,features from X,y
      size_t n_classes = size_t(y.max())- size_t(y.min())+1; 

      size_t n_features = 0;
      if(X.size1() > 0 && X.size2() > 0)
      {
        n_features = X.size2(); 
      }
      else
      {
        std::cout<<"[-] error loading X : "<<X_file<<std::endl; 
        continue; 
      }
        

       
      //create a super-tree from the datasets and fit X to y
      std::cout<<"[+] creating super-tree from dataset : "<<dat<<std::endl; 
      //print the number of classes and features 
      std::cout<<"[+] n_classes : "<<n_classes<<std::endl; 
      std::cout<<"[+] n_features : "<<n_features<<std::endl;
      auto time_start = std::chrono::high_resolution_clock::now(); 


      std::vector<size_t> y_vector(y.begin(),y.end()); 

      provallo::super_tree<real_t,size_t> hyper_matrix(X,y_vector,X.size2(),y.size2());
      //configure hyper_matrix 
      hyper_matrix.set_eta(0.1);
      hyper_matrix.set_momentum(0.9);
      //
 
      //fit X to y
      std::cout<<"[+] fitting X to y : "<<dat<<std::endl;


      //transform y to a vector of size_t 
      hyper_matrix.fit(X,y_vector);
      
      auto time_end = std::chrono::high_resolution_clock::now(); 
      auto elapsed_fit_time = std::chrono::duration_cast<std::chrono::milliseconds>(time_end-time_start); 

      std::cout<<"[+] fit time : "<<std::to_string(elapsed_fit_time.count())<<std::endl; 
      auto stability = hyper_matrix.stability();
      auto divergence = hyper_matrix.divergent_stability(); 
      auto flutter = hyper_matrix.flutter_stability();
      auto unstable = hyper_matrix.unstable_stability();
    
      std::cout<<"[+] stability : "<<std::to_string(stability)<<std::endl;
      std::cout<<"[+] divergence : "<<std::to_string(divergence)<<std::endl;
      std::cout<<"[+] flutter : "<<std::to_string(flutter)<<std::endl;
      std::cout<<"[+] unstable : "<<std::to_string(unstable)<<std::endl;



      //save the decision function to file in a comma separated format 
      auto score = hyper_matrix.score(X,y_vector);
      std::cout<<"[+] saving train score to file : "<<dat<<std::endl; 
      std::string decision_function_file = std::string("./df/")+dat+std::string("_decision_function.csv"); 
      std::ofstream decision_function(decision_function_file); 
      //save the score to file 

      for(size_t i=0;i<score.size()-1;i++)
      {
        decision_function<<std::to_string(score[i])<<",";
      }
      decision_function<<std::to_string(  score[score.size()-1])<<std::endl; 

      decision_function.close();
      
      //measure predict time 
      auto start_time = std::chrono::high_resolution_clock::now(); 

      std::vector<size_t> yy = hyper_matrix.predict(X); 
      auto end_time = std::chrono::high_resolution_clock::now(); 
      auto elapsed_predict_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time-start_time); 
      std::cout<<"[+] predict time : "<<std::to_string(elapsed_predict_time.count())<<std::endl; 

      //save the predictions to file 
      std::cout<<"[+] saving predictions to file : "<<dat<<std::endl; 
      std::string predictions_file = std::string("./pred/")+dat+std::string("_predictions.csv"); 
      std::ofstream predictions(predictions_file); 
      //save yy to file
      for(size_t i=0;i<yy.size()-1;i++)
      {
        predictions<<std::to_string(yy[i])<<",";
      }
      predictions<<std::to_string(yy[yy.size()-1])<<std::endl;
      predictions.close();

      //compare yy to y
      std::cout<<"[+] comparing yy to y : "<<dat<<std::endl; 
      size_t correct_predictions = 0;  
      real_t accuracy = 0.,precision=0.,recall=0.,f1_score=0.; 
      real_t true_positives=0.,true_negatives=0.,false_positives=0.,false_negatives=0.; 
      real_t anomaly_ratio = 0.0; 

      for(size_t i=0;i<yy.size();i++)
      {
        //if prediction is out of class boundaries, it's an anomaly 
        if(yy[i] < y.min() || yy[i] > y.max())
        {         
          std::cout<<"[+] anomaly detected on sample index : "<<std::to_string(i)<<std::endl; 
          anomaly_ratio+=1.0/(real_t)yy.size(); 

        }

        if(yy[i] == y(0,i))
        {
          correct_predictions++; 
          if(yy[i] == 1)
          {
            true_positives++; 
          }
          else
          {
            true_negatives++; 
          }
        }
        else
        {
          if(yy[i] == 1)
          {
            false_positives++; 
          }
          else
          {
            false_negatives++; 
          }
        }
      } 

      //calculate accuracy
      accuracy = (real_t)correct_predictions/(real_t)yy.size();
      //calculate precision
      precision =(false_positives>0)? (real_t)true_positives/(real_t)(true_positives+false_positives):1.;
      //calculate recall
      recall = (real_t)true_positives/(real_t)(true_positives+false_negatives);
      //calculate f1_score
      f1_score = (precision+recall>0)? 2.0*(precision*recall)/(precision+recall):1.;

      
      //accomulate auc roc 
      //calculate fpr,tpr,roc_areas :
      //fpr = false_positives/(false_positives+true_negatives) 
      //print accuracy

      std::cout<<"[+] accuracy : "<<accuracy<<std::endl; 
      std::cout<<"[+] precision : "<<precision<<std::endl;
      std::cout<<"[+] recall : "<<recall<<std::endl;
      std::cout<<"[+] f1_score : "<<f1_score<<std::endl;
      std::cout<<"[+] eta : "<<hyper_matrix.get_eta()<<std::endl; 
      std::cout<<"[+] momentum : "<<hyper_matrix.get_momentum()<<std::endl; 
      std::cout<<"[+] anomaly ratio : "<<std::to_string(anomaly_ratio)<<std::endl; 
      
      //calculate fpr,tpr,roc_areas :
      //create a vectorizer from the super-tree
      //check if the super  is valid 
      //save results to file 
 
    }
    std::getchar();
    
 }
 ```



With <3 from San Francisco
