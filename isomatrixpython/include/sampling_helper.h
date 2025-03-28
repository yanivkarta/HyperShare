#ifndef __SAMPLING_HELPER_H_
#define __SAMPLING_HELPER_H_

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <cmath>
#include <numeric>
#include <complex>
#include <valarray>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <ctime>

#include "matrix.h"
#include "optimizers.h"

//implement FFT for sampling


namespace provallo
{
    template<typename T>
    class sampling_helper
    {
        public:
            sampling_helper() = default;
            ~sampling_helper() = default;

            static void
            fft(std::valarray<std::complex<T>> &x)
            {
                const size_t N = x.size();
                if (N <= 1) return;

                // divide
                std::valarray<std::complex<T>> even = x[std::slice(0, N/2, 2)];
                std::valarray<std::complex<T>>  odd = x[std::slice(1, N/2, 2)];

                // conquer
                fft(even);
                fft(odd);

                // combine
                for (size_t k = 0; k < N/2; ++k)
                {
                    std::complex<T> t = std::polar(1.0, -2 * M_PI * k / N) * odd[k];
                    x[k    ] = even[k] + t;
                    x[k+N/2] = even[k] - t;
                }
                
                x=x.apply(std::conj); 
                //finalize

                
            }
            static void  fft ( std::complex<T> *x , size_t N)
            {
                if (N <= 1) return;

                // divide
                std::valarray<std::complex<T>> even =  std::valarray<std::complex<T>>(x,N)[std::slice(0, N/2, 2)];
                std::valarray<std::complex<T>>  odd =  std::valarray<std::complex<T>>(x,N)[std::slice(1, N/2, 2)];

                // conquer
                fft(even);
                fft(odd);

                // combine
                for (size_t k = 0; k < N/2; ++k)
                {
                    std::complex<T> t = std::polar(1.0, -2 * M_PI * k / N) * odd[k];
                    x[k    ] = even[k] + t;
                    x[k+N/2] = even[k] - t;
                }
            }
            static void
            ifft(std::valarray<std::complex<double>> &x)
            {
                // conjugate the complex numbers
                x = x.apply(std::conj);

                // forward fft
                fft( x );

                // conjugate the complex numbers again
                x = x.apply(std::conj);

                // scale the numbers
                x /= x.size();
            }

            static  std::valarray<std::complex<double>>
            fft(const std::valarray<std::complex<double>> &x)
            {
                std::valarray<std::complex<double>> y = x;
                fft(y);
                return y;
            }

            static  std::valarray<std::complex<double>>
            ifft(const std::valarray<std::complex<double>> &x)
            {
                std::valarray<std::complex<double>> y = x;
                ifft(y);
                return y;
            }
            //fft of matrix 
            static matrix<std::complex<double>>
            fft(const matrix<double> &x)
            {
                matrix<std::complex<double>> y(x.rows(),x.cols());
                for(size_t i = 0 ; i < x.rows() ; ++i)
                {
                    for(size_t j = 0 ; j < x.cols() ; ++j)
                    {
                        y(i,j) = x(i,j);

                    }
                    //fft of row
                    fft(y[i] , y.cols());

                }
                return y;
            }   
            //ifft of matrix
            static matrix<std::complex<double>>
            ifft(const matrix<std::complex<double>> &x)
            {
                matrix<std::complex<double>> y(x.rows(),x.cols());
                for(size_t i = 0 ; i < x.rows() ; ++i)
                {
                    for(size_t j = 0 ; j < x.cols() ; ++j)
                    {
                        y(i,j) = x(i,j);
                    }
                }
                return y;
            }
    };
    
    template<typename T>

    class fft_sampling_helper
    {
        //converts from matrix<T> to matrix<std::complex<double>> and vice versa 
        matrix<std::complex<T>> fft_matrix;
        public:
            fft_sampling_helper() = default;
            ~fft_sampling_helper() = default;

            static void
            fft(matrix<T>& x)
            {
                x = sampling_helper<T>::fft(x);
            }
            static void
            ifft(matrix<T>& x)
            {
                x = sampling_helper<T>::ifft(x);
            }
            matrix<std::complex<double>> operator()(matrix<T>& x)
            {
                return sampling_helper<T>::fft(x);
            }
            matrix<T> operator()(matrix<std::complex<T>>& x)
            {
                return sampling_helper<T>::ifft(x);
            }
            //initialize the fft matrix
            matrix<std::complex<T>> operator()(size_t rows , size_t cols)
            {
                return fft_matrix;
            }

            void init( size_t cols)
            {
                fft_matrix = matrix<std::complex<double>>(cols,cols);
                for(size_t i = 0 ; i < cols ; ++i)
                {
                    for(size_t j = 0 ; j < cols ; ++j)
                    {
                        fft_matrix(i,j) = std::polar(1.0, -2 * M_PI * i * j / cols);
                    }
                }

            }
            
    };

    enum activation_helper_activations 
    {
        sigmoid_default,
        sigmoid_derivative,
        tanh,
        tanh_derivative,
        relu,
        relu_derivative,
        softmax_default
    };
    
    template<typename T>
    class activation_helper
    {
        std::function<void(provallo::matrix<T>&)> activation_function;

        public:

            activation_helper(activation_helper_activations activation = activation_helper_activations::sigmoid_default )
            {
                switch(activation)
                {
                    case activation_helper_activations::sigmoid_default:
                        activation_function = sigmoid;
                        break;
                    case activation_helper_activations::sigmoid_derivative:
                        activation_function = sigmoid_derivative;
                        break;
                    case activation_helper_activations::tanh:
                        activation_function = tanh;
                        break;
                    case activation_helper_activations::tanh_derivative:
                        activation_function = tanh_derivative;
                        break;
                    case activation_helper_activations::relu:
                        activation_function = relu;
                        break;
                    case activation_helper_activations::relu_derivative:
                        activation_function = relu_derivative;
                        break;
                    case activation_helper_activations::softmax_default:
                        activation_function = softmax;
                        break;
                    default:
                        activation_function = sigmoid;
                        break;
                }
            }
            ~activation_helper() = default;

            matrix<T> operator ()(matrix<T>& x)
            {
                if( activation_function == nullptr )
                activation_function = sigmoid;
                
                matrix<T> r(x);                
                activation_function(r);
                return r;
            }

            static void
            sigmoid(matrix<T>& x)
            {
                x = 1.0 / ( matrix<T>(-x).exp()+1.);
            }
            static void
            sigmoid_derivative(matrix<T>& x)
            {
                x = x * (1.0 - x);
            }
            static void
            tanh(matrix<T>& x)
            {
                x = x.tanh();
            }
            static void
            tanh_derivative(matrix<T>& x)
            {
                x = 1.- x.pow(2);
            }
            static void
            relu(matrix<T>& x)
            {
                x =x*x.max();
            }
            static void
            relu_derivative(matrix<T>& x)
            {
                x = x.unaryExpr([](T x) { return x > 0 ? 1 : 0; });
            }
            static void
            softmax(matrix<T>& x)
            {
                x = (x - x.max()).exp();
                x = x / x.sum();
            }
            static void
            softmax_derivative(matrix<T>& x)
            {
                x *= (1.0 - x) * x;
                
            }

            //

    };

    template<typename T>
    class rms_prop_helper
    {
        public:
            rms_prop_helper() = default;
            ~rms_prop_helper() = default;

            static void
            rms_prop(matrix<T>& weights, 
                     matrix<T>& gradients,
                     matrix<T>& rms,
                     real_t learning_rate,
                     real_t decay_rate)
            {
                //rms = decay_rate * rms + (1 - decay_rate) * gradients^2
                rms = decay_rate * rms + (1 - decay_rate) * gradients.pow(2);
                //weights = weights - learning_rate * gradients / sqrt(rms + 1e-5)
                weights = weights - learning_rate * gradients / (rms + 1e-5).sqrt();

                //
            }
            //rms prop operator return a new matrix instance of weights
            matrix<T> operator()(matrix<T>& weights, 
                                 matrix<T>& gradients,
                                 matrix<T>& rms,
                                 real_t learning_rate, 
                                 real_t decay_rate)
            {
                rms_prop(weights, gradients, rms, learning_rate, decay_rate);
                return weights;
            }
            

    };
    template<typename T>

    class adam_helper 
    {
        public:
            adam_helper() = default;
            ~adam_helper() = default;

            static void
            adam(matrix<T>& weights, 
                 matrix<T>& gradients,
                 matrix<T>& m,
                 matrix<T>& v,
                 real_t learning_rate,
                 real_t beta1,
                 real_t beta2,
                 real_t epsilon)
            {
                //m = beta1 * m + (1 - beta1) * gradients
                m = beta1 * m + (1 - beta1) * gradients;
                //v = beta2 * v + (1 - beta2) * gradients^2
                v = beta2 * v + (1 - beta2) * gradients.pow(2);
                //weights = weights - learning_rate * m / (sqrt(v) + epsilon)
                weights = weights - learning_rate * m / (v.sqrt() + epsilon);
            }
            //adam operator return a new matrix instance of weights 
            matrix<T> operator()(matrix<T>& weights, 
                                 matrix<T>& gradients,
                                 matrix<T>& m,
                                 matrix<T>& v,
                                 real_t learning_rate,
                                 real_t beta1,
                                 real_t beta2,
                                 real_t epsilon)
            {
                adam(weights, gradients, m, v, learning_rate, beta1, beta2, epsilon);
                return weights;
            }   

    };
    template<typename T>
    class adagrad_helper
    {
        public:
            adagrad_helper() = default;
            ~adagrad_helper() = default;

            static void
            adagrad(matrix<T>& weights, 
                    matrix<T>& gradients,
                    matrix<T>& cache,
                    real_t learning_rate,
                    real_t epsilon)
            {
                //cache = cache + gradients^2
                cache = cache + gradients.pow(2);
                //weights = weights - learning_rate * gradients / (sqrt(cache) + epsilon)
                weights = weights - learning_rate * gradients / (cache.sqrt() + epsilon);
            }
            //adagrad operator return a new matrix instance of weights
            matrix<T> operator()(matrix<T>& weights, 
                                 matrix<T>& gradients,
                                 matrix<T>& cache,
                                 real_t learning_rate,
                                 real_t epsilon)
            {
                adagrad(weights, gradients, cache, learning_rate, epsilon);
                return weights;
            }
    };  
    template<typename T>
    class adadelta_helper
    {
        public:
            adadelta_helper() = default;
            ~adadelta_helper() = default;

            static void
            adadelta(matrix<T>& weights, 
                     matrix<T>& gradients,
                     matrix<T>& cache,
                     matrix<T>& delta,
                     double learning_rate,
                     double decay_rate,
                     double epsilon)
            {
                //cache = decay_rate * cache + (1 - decay_rate) * gradients^2
                cache = decay_rate * cache + (1 - decay_rate) * gradients.pow(2);
                //delta = sqrt(delta + epsilon) / sqrt(cache + epsilon) * gradients
                delta = (delta + epsilon).sqrt() / (cache + epsilon).sqrt() * gradients;
                //weights = weights - learning_rate * delta
                weights = weights - learning_rate * delta;
            }

            //adadelta operator return a new matrix instance of weights
            matrix<T> operator()(matrix<T>& weights, 
                                 matrix<T>& gradients,
                                 matrix<T>& cache,
                                 matrix<T>& delta,
                                 double learning_rate,
                                 double decay_rate,
                                 double epsilon)
            {

                adadelta(weights, gradients, cache, delta, learning_rate, decay_rate, epsilon);
                return weights;
            }
    };  
    template<typename T>

    class momentum_helper
    {
        public:
            momentum_helper() = default;
            ~momentum_helper() = default;

            static void
            momentum(matrix<T>& weights, 
                     matrix<T>& gradients,
                     matrix<T>& velocity,
                     double learning_rate,
                     double momentum)
            {
                //velocity = momentum * velocity - learning_rate * gradients
                velocity = momentum * velocity - learning_rate * gradients;
                //weights = weights + velocity
                weights = weights + velocity;
            }
            //momentum operator return a new matrix instance of weights
            matrix<T> operator()(matrix<T>& weights, 
                                 matrix<T>& gradients,
                                 matrix<T>& velocity,
                                 double learning_rate,
                                 double momentum)
            {
                momentum(weights, gradients, velocity, learning_rate, momentum);
                return weights;
            }   
    };  
    template<typename T>
    class nesterov_helper
    {
        public:
            nesterov_helper() = default;
            ~nesterov_helper() = default;

            static void
            nesterov(matrix<T>& weights, 
                     matrix<T>& gradients,
                     matrix<T>& velocity,
                     real_t learning_rate,
                     real_t momentum)
            {
                //velocity = momentum * velocity - learning_rate * gradients
                velocity = momentum * velocity - learning_rate * gradients;
                //weights = weights + momentum * velocity - learning_rate * gradients
                weights = weights + momentum * velocity - learning_rate * gradients;
            }
            //nesterov operator return a new matrix instance of weights
            matrix<T> operator()(matrix<T>& weights, 
                                 matrix<T>& gradients,
                                 matrix<T>& velocity,
                                 real_t learning_rate,
                                 real_t momentum)
            {
                
                nesterov(weights, gradients, velocity, learning_rate, momentum);
                return weights;
            }
    };
    template<typename T>
    class sgd_helper
    {
        public:
            sgd_helper() = default;
            ~sgd_helper() = default;

            static void
            sgd(matrix<T>& weights, 
                matrix<T>& gradients,
                real_t learning_rate)
            {
                //weights = weights - learning_rate * gradients
                weights = weights - learning_rate * gradients;
            }
            //sgd operator return a new matrix instance of weights
            matrix<T> operator()(matrix<T>& weights, 
                                 matrix<T>& gradients,
                                 real_t learning_rate)
            {
                sgd(weights, gradients, learning_rate);
                return weights;
            }

    };
        template<typename T>

    class sgd_momentum_helper
    {
        public:
            sgd_momentum_helper() = default;
            ~sgd_momentum_helper() = default;

            static void
            sgd_momentum(matrix<T>& weights, 
                         matrix<T>& gradients,
                         matrix<T>& velocity,
                         real_t learning_rate,
                         real_t momentum)
            {
                //velocity = momentum * velocity - learning_rate * gradients
                velocity = momentum * velocity - learning_rate * gradients;
                //weights = weights + velocity
                weights = weights + velocity;
            }
            //sgd_momentum operator return a new matrix instance of weights
            matrix<T> operator()(matrix<T>& weights, 
                                 matrix<T>& gradients,
                                 matrix<T>& velocity,
                                 real_t learning_rate,
                                 real_t momentum)
            {
                sgd_momentum(weights, gradients, velocity, learning_rate, momentum);
                return weights;
            }   
    };
    
    template<typename T>
    class sgd_nesterov_helper
    {
        public:
            sgd_nesterov_helper() = default;
            ~sgd_nesterov_helper() = default;

            static void
            sgd_nesterov(matrix<T>& weights, 
                         matrix<T>& gradients,
                         matrix<T>& velocity,
                         real_t learning_rate,
                         real_t momentum)
            {
                //gradient descent with momentum and Nesterov acceleration 

                //calculate the momentum
                matrix<T> v = momentum * velocity;
                //velocity = momentum * velocity - learning_rate * gradients
                velocity = v - learning_rate * gradients;
                //weights = weights + momentum * velocity - learning_rate * gradients
                weights = weights + v - learning_rate * gradients;
            }
            //sgd_nesterov operator return a new matrix instance of weights
            matrix<T> operator()(matrix<T>& weights, 
                                 matrix<T>& gradients,
                                 matrix<T>& velocity,
                                 real_t learning_rate,
                                 real_t momentum)
            {
                sgd_nesterov(weights, gradients, velocity, learning_rate, momentum);
                return weights;
            }

            
    };
    //adamax_helper:
    template<typename T>
    class  adamax_helper
    {
        //adamax operator return a new matrix instance of weights
        matrix<T> operator()(matrix<T>& weights, 
                             matrix<T>& gradients,
                             matrix<T>& velocity,
                             real_t learning_rate,
                             real_t momentum)
        {
                //velocity = momentum * velocity - learning_rate * gradients
                velocity = momentum * velocity - learning_rate * gradients;
                //weights = weights + velocity
                weights = weights + velocity;
                return weights;
        }
        

    };

    //nadam helper
    template <typename T> 
    class nadam_helper : public adamax_helper<T>
    {
        //static void nadam(matrix<T>& weights,
        //                  matrix<T>& gradients,
        //                  matrix<T>& velocity,
        //                  real_t learning_rate,
        //                  real_t momentum)
        //{
        //    //velocity = momentum * velocity - learning_rate * gradients
        //    velocity = momentum * velocity - learning_rate * gradients;
        //    //weights = weights + velocity
        //    weights = weights + velocity;
        //}
        //nadam operator return a new matrix instance of weights
        matrix<T> operator()(matrix<T>& weights, 
                             matrix<T>& gradients,
                             matrix<T>& velocity,
                             real_t learning_rate,
                             real_t momentum)
        {
                //velocity = momentum * velocity - learning_rate * gradients
                velocity = momentum * velocity - learning_rate * gradients;
                //weights = weights + velocity
                weights = weights + velocity;
                return weights;
        }   
        

    };


    //amsgrad_helper:
    template<typename T>
    class amsgrad_helper
    {
        //amsgrad operator return a new matrix instance of weights
        matrix<T> operator()(matrix<T>& weights, 
                             matrix<T>& gradients,
                             matrix<T>& velocity,
                             real_t learning_rate,
                             real_t momentum)
        {
                //velocity = momentum * velocity - learning_rate * gradients
                velocity = momentum * velocity - learning_rate * gradients;
                //weights = weights + velocity
                weights = weights + velocity;
                return weights;
        }

    };

    
    template <typename T>
    class sampling_filter
    {
        real_t epsilon = 1e-6; 
        real_t learning_rate = 1e-3;
        real_t momentum = 0.9;
        real_t beta = 0.5;

        matrix<T> weights;
        matrix<T> gradients;
        matrix<T> cache;
        matrix<T> delta;
        matrix<T> velocity;
        enum sampling_type
        {
            NO_SAMPLING,
            NESTEROV,
            SGD,
            SGD_MOMENTUM,
            SGD_NESTEROV
        }; 
        //default sampling
        enum sampling_type type = NONE;
        
        enum filter_type : int8_t
        {
            NONE,
            SAMPLING,
            ADAGRAD,
            ADAM,
            ADAMAX,
            AMSGRAD,
            NADAM,  
            SAMPLING_FILTER // use the vector of filters
        };
        enum filter_type filter_type = SAMPLING_FILTER; 

        std::vector<enum filter_type> filters;

        public:
            
            //constructors:
            sampling_filter( sampling_type ty = NO_SAMPLING) : type(ty) {} 
            ~sampling_filter() = default;
            matrix<T> sample(matrix<T>& weights, 
                             matrix<T>& gradients)
            {
                //update the gradients:
                gradients = gradients + epsilon;
                //update the cache:
                cache = cache + gradients.pow(2);
                //update the delta:
                delta = beta * delta + (1 - beta) * gradients.pow(2);
                //update the velocity:
                velocity = momentum * velocity - learning_rate * gradients / (cache.sqrt() + epsilon);
                //update the weights:
                weights = weights + velocity;

                //apply the sampling
                switch (type)
                {
                    case SAMPLING:
                        sampling_helper<T> helper;
                        weights = helper(weights, gradients, epsilon, learning_rate, momentum);
                        break;
                    case SGD:
                        sgd_helper<T> helper_sgd;
                        weights = helper_sgd(weights, gradients, learning_rate);
                        break;
                    case SGD_MOMENTUM:
                        sgd_momentum_helper<T> helper_sgd_momentum;
                        weights = helper_sgd_momentum(weights, gradients, velocity, learning_rate, momentum);
                        break;
                    case SGD_NESTEROV:
                        sgd_nesterov_helper<T> helper_sgd_nesterov;
                        weights = helper_sgd_nesterov(weights, gradients, velocity, learning_rate, momentum);
                        break;
                    default:
                        break;

                }

                return weights;

            }
            matrix<T> filter(matrix<T>& weights, 
                             matrix<T>& gradients)
            {
                
                
                //update the gradients:
                gradients = gradients + epsilon;
                //update the cache:
                cache = cache + gradients.pow(2);
                //update the delta:
                delta = beta * delta + (1 - beta) * gradients.pow(2);
                //update the velocity:
                velocity = momentum * velocity - learning_rate * gradients / (cache.sqrt() + epsilon);
                //update the weights:
                weights = weights + velocity;
                if(filters.size() == 0)
                {
                    switch (filter_type)
                    {
                        case SAMPLING_FILTER:
                            sampling_helper<T> helper;
                            weights = helper(weights, gradients, epsilon, learning_rate, momentum);
                            break;
                        case ADAGRAD:
                            adagrad_helper<T> helper_adagrad;
                            weights = helper_adagrad(weights, gradients, cache, learning_rate, epsilon);
                            break;
                        case ADAM:
                            adam_helper<T> helper_adam;
                            weights = helper_adam(weights, gradients, cache, delta, learning_rate, epsilon);
                            break;
                        case ADAMAX:
                            adamax_helper<T> helper_adamax;
                            weights = helper_adamax(weights, gradients, cache, delta, learning_rate, epsilon);
                            break;
                        case AMSGRAD:
                            amsgrad_helper<T> helper_amsgrad;
                            weights = helper_amsgrad(weights, gradients, cache, delta, learning_rate, epsilon);
                            break;
                        case NADAM:
                            nadam_helper<T> helper_nadam;
                            weights = helper_nadam(weights, gradients, cache, delta, learning_rate, epsilon);
                            break;

                        default:
                            //do nothing
                            break;
                    }
                    //do nothing
                }
                //apply the filters
                for (size_t i = 0; i < filters.size(); ++i)
                {
                    switch (filters[i])
                    {
                        case SAMPLING:
                            sampling_helper<T> helper;
                            weights = helper(weights, gradients, epsilon, learning_rate, momentum);
                            break;
                        case SGD:
                            sgd_helper<T> helper_sgd;
                            weights = helper_sgd(weights, gradients, learning_rate);
                            break;
                        case SGD_MOMENTUM:
                            sgd_momentum_helper<T> helper_sgd_momentum;     
                            weights = helper_sgd_momentum(weights, gradients, velocity, learning_rate, momentum);
                            break;
                        case SGD_NESTEROV:
                            sgd_nesterov_helper<T> helper_sgd_nesterov;
                            weights = helper_sgd_nesterov(weights, gradients, velocity, learning_rate, momentum);
                            break;
                        default:
                            //do nothing
                            break;

                    }
                }//end for
                
                
                
                
                return weights;

            }
     };
}//end namespace provallo


#endif//__SAMPLING_HELPER_H_