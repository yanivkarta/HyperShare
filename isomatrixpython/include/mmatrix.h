#include "matrix.h"

using namespace std;
using namespace provallo;

#define MAX_TENSOR_SIZE 1000000ul

template <class T>
class spinor
{
    // storage of spinor
    matrix<T> m;
    std::vector<T> spinor;
    // constructor
public:
    spinor<T>(const matrix<T>& m) : m(m), spinor(m.size1() * m.size2())
    {
        // construct spinor according to m
        std::copy(m.data(), m.data() + m.size1() * m.size2(), spinor.begin());
    }
    //copy constructor
    spinor<T>(const spinor<T>& other) : m(other.m), spinor(other.spinor) {} 
    
    //convert to vector
    operator std::vector<T>() const { return spinor; }

    // operate on the spinor using matrix multiplication
    spinor<T> operator*(const matrix<T>& other) const {
        matrix<T> temp = m * other;
        return spinor<T>(temp);
    }

    spinor operator*(const spinor& other) const {
        matrix<T> temp = m * other.m;
        return spinor<T>(temp);
    }

    // add rotation operation
    spinor<T> rotate_transform(const matrix<T>& rotationMatrix) const {
        matrix<T> temp = rotationMatrix.T() * m;
        return spinor<T>(temp);
    } 
    spinor<T> rotate(const matrix<T>& rotationMatrix) const {
        matrix<T> temp = rotationMatrix * m;
        return spinor<T>(temp);
    }
    //ladder operator
    spinor<T> operator+(const spinor<T>& other) const {
        matrix<T> temp = m + other.m;
        return spinor<T>(temp);
    }   

    spinor<T> operator-(const spinor<T>& other) const {
        matrix<T> temp = m - other.m;
        return spinor<T>(temp);
    }

    spinor<T> operator*(const T& other) const {
        matrix<T> temp = m * other;
        return spinor<T>(temp);
    }   

    spinor<T> operator/(const T& other) const {
        matrix<T> temp = m / other;
        return spinor<T>(temp);
    }   

    //get the norm of the spinor
    T norm() const{
        return m.norm();
    }

    //get the squared norm of the spinor
    T squaredNorm() const{
        return m.squaredNorm();
    }
    //get the conjugate of the spinor
    //the matrix conjugate already returns the real part
    //conj returns the complex conjugate
    spinor<std::complex<T>> conj() const{

        return spinor<std::complex<T>>(m.complex_conjugate());
    }
    spinor<std::complex<T>> sqrt() const{
        return spinor<std::complex<T>>(m.sqrt()); 
    }
    spinor<T> angular_momentum(T momentum) const{

        //calculate the angular momentum 
        //angular momentum working on the spinor matrix 
        //would multiply the elements of the matrix by the cosine and sine of the momentum according to the spinor definition of the angular momentum operator 
        matrix<T> temp = m * momentum;
        for (size_t i = 0; i < temp.size1(); i++){
            for (size_t j = 0; j < temp.size2(); j++){
                temp(i, j) = std::cos(momentum) * temp(i, j) - std::sin(momentum) * temp(i, j + 1);
                temp(i, j + 1) = std::sin(momentum) * temp(i, j) + std::cos(momentum) * temp(i, j + 1);
            }
        }   
        return spinor<T>(temp);
    }
    //complex plane momentum operator
    spinor<std::complex<T>> momentum(T momentum) const{
        //calculate the momentum 
        //momentum working on the spinor matrix 
        //would multiply the elements of the matrix by the cosine and sine of the momentum according to the spinor definition of the momentum operator 
        matrix<std::complex<T>> temp = m * momentum;
        for (size_t i = 0; i < temp.size1(); i++){
            for (size_t j = 0; j < temp.size2(); j++){
                temp(i, j) = std::cos(momentum) * temp(i, j) - std::sin(momentum) * temp(i, j + 1);
                temp(i, j + 1) = std::sin(momentum) * temp(i, j) + std::cos(momentum) * temp(i, j + 1);
            }
        }   
        return spinor<std::complex<T>>(temp);
    } 

    //partial deriviative with a theta and phi angle
    spinor<T> partial_theta(T theta, T phi) const{

        matrix<T> temp = m * theta; 
        for (size_t i = 0; i < temp.size1(); i++){

            for (size_t j = 0; j < temp.size2(); j++){
                
                temp(i, j) = std::tan(theta) * temp(i, j) * (std::cos(theta) / std::sin(phi));   
            }
        }    
        return spinor<T>(temp);

    }

    virtual ~spinor() {}

}; //end of spinor

//tensor
template <class T>
class tensor : public matrix<T>
{
    // storage of tensor
    matrix<T> m1, m2, m3;
    std::vector<T> tensor;

public:
    tensor<T>(const matrix<T>& m1, const matrix<T>& m2, const matrix<T>& m3) : m1(m1), m2(m2), m3(m3), tensor(m1.size1() * m1.size2() * m2.size2() * m3.size2())
    {
        // construct tensor according to m1, m2 and m3
        for (size_t i = 0; i < m1.size1(); i++)
        {
            for (size_t j = 0; j < m1.size2(); j++)
            {
                for (size_t k = 0; k < m2.size2(); k++)
                {
                    for (size_t l = 0; l < m3.size2(); l++)
                    {
                        tensor[i * m1.size2() * m2.size2() * m3.size2() + j * m2.size2() * m3.size2() + k * m3.size2() + l] = m1(i, j) * m2(j, k) * m3(k, l);
                    }
                }
            }
        }
    }
    
    tensor<T>(const std::vector<T>& tensor) : tensor(tensor) {

        //initialize m1, m2 and m3 from tensor
        auto size = sqrt(tensor.size());
        m1.resize(size, size);
        m2.resize(size, size);
        m3.resize(size, size);


        for (size_t i = 0; i < m1.size1(); i++)
        {
            for (size_t j = 0; j < m1.size2(); j++)
            {
                for (size_t k = 0; k < m2.size2(); k++)
                {
                    for (size_t l = 0; l < m3.size2(); l++)
                    {
                        m1(i, j) = tensor[i * m1.size2() * m2.size2() * m3.size2() + j * m2.size2() * m3.size2() + k * m3.size2() + l];
                        m2(j, k) = tensor[i * m1.size2() * m2.size2() * m3.size2() + j * m2.size2() * m3.size2() + k * m3.size2() + l];
                        m3(k, l) = tensor[i * m1.size2() * m2.size2() * m3.size2() + j * m2.size2() * m3.size2() + k * m3.size2() + l]; 

                    }
                }
            }
        }   
         
    } 

    
    tensor<T>(const matrix<T>& m1, const matrix<T>& m2, const matrix<T>& m3, const std::vector<T>& tensor) : m1(m1), m2(m2), m3(m3), tensor(tensor) {} 


    



    operator std::vector<T>() const { return tensor; }
    //operate on the tensor elements using operator overloading 
    const T& operator()(size_t i, size_t j, size_t k, size_t l) const { return tensor[i * m1.size2() * m2.size2() * m3.size2() + j * m2.size2() * m3.size2() + k * m3.size2() + l]; } 

    // operate on the tensor using matrix multiplication
    tensor<T> operator*(const matrix<T>& other) const {
        return tensor<T>(m1 * other, m2, m3);
    }

    tensor operator*(const tensor& other) const {
        return tensor<T>(m1 * other.m1, m2 * other.m2, m3 * other.m3);
    }
    ~tensor() {}

    //add rotation operation
    tensor<T> rotate_transform(const matrix<T>& rotationMatrix) const {
        return tensor<T>(rotationMatrix.T() * m1, rotationMatrix.T() * m2, rotationMatrix.T() * m3);
    }
    //ladder operator
    tensor<T> operator+(const tensor<T>& other) const {
        return tensor<T>(m1 + other.m1, m2 + other.m2, m3 + other.m3);
    }

    tensor<T> operator-(const tensor<T>& other) const {
        return tensor<T>(m1 - other.m1, m2 - other.m2, m3 - other.m3);
    }

    tensor<T> operator*(const T& other) const {
        return tensor<T>(m1 * other, m2 * other, m3 * other);
    }

    tensor<T> operator/(const T& other) const {
        return tensor<T>(m1 / other, m2 / other, m3 / other);
    }
    
    //angular momentum operator for tensor
    tensor<T> operator^(const tensor<T>& other) const {
        return tensor<T>(m1 * other.m1, m2 * other.m2, m3 * other.m3);
    }

    //get the norm of the tensor
    T norm() const{
        return m1.norm() * m2.norm() * m3.norm();
    }

    //get the squared norm of the tensor
    T squaredNorm() const{
        return m1.squaredNorm() * m2.squaredNorm() * m3.squaredNorm();
    }

    //get the conjugate of the tensor
    tensor<std::complex<T>> conj() const{
        return tensor<std::complex<T>>(m1.complex_conjugate(), m2.complex_conjugate(), m3.complex_conjugate());
    }
    //get the transpose of the tensor
    tensor<T> T() const{
        return tensor<T>(m1.T(), m2.T(), m3.T());
    }   
    //get the trace of the tensor
    T trace() const{
        return m1.trace() * m2.trace() * m3.trace();
    }
    //get the determinant of the tensor
    T det() const{
        return m1.det() * m2.det() * m3.det();
    }
    //get the inverse of the tensor
    tensor<T> inverse() const{
        return tensor<T>(m1.inverse(), m2.inverse(), m3.inverse());
    }
    //get the adjoint of the tensor
    tensor<T> adjoint() const{
        return tensor<T>(m1.adjoint(), m2.adjoint(), m3.adjoint());
    }
    //get the eigenvalues of the tensor
    tensor<T> eigenvalues() const{
        return tensor<T>(m1.eigenvalues(), m2.eigenvalues(), m3.eigenvalues());
    }
    //get the eigenvectors of the tensor
    tensor<T> eigenvectors() const{
        return tensor<T>(m1.eigenvectors(), m2.eigenvectors(), m3.eigenvectors());
    }   
    const size_t size() const { return tensor.size(); }
    const T* data() const { return tensor.data(); }
    T* data() { return tensor.data(); }

    //get the eigenvalues and eigenvectors of the tensor
    void get_eigen_values_and_vectors(std::vector<T> &eigen_values, tensor<T> &eigen_vectors) const{
        m1.get_eigen_values_and_vectors(eigen_values, eigen_vectors.m1);
        m2.get_eigen_values_and_vectors(eigen_values, eigen_vectors.m2);
        m3.get_eigen_values_and_vectors(eigen_values, eigen_vectors.m3);
    }   

};


//N-dimensional spin tensor of matrixes 
template <typename T>
tensor<T> operator*(const tensor<T>& a, const tensor<T>& b) {
    return tensor<T>(a * b.m1, a * b.m2, a * b.m3);
}

//N-dimensional spin tensor of matrixes 
template <typename T>
tensor<T> operator+(const tensor<T>& a, const tensor<T>& b) {
    return tensor<T>(a + b.m1, a + b.m2, a + b.m3);
}
//N-dimensional spin tensor of matrixes 
template <typename T>
tensor<T> operator-(const tensor<T>& a, const tensor<T>& b) {
    return tensor<T>(a - b.m1, a - b.m2, a - b.m3);
}   
//N-dimensional spin tensor of matrixes 
template <typename T>
tensor<T> operator/(const tensor<T>& a, const tensor<T>& b) {
    return tensor<T>(a / b.m1, a / b.m2, a / b.m3);
}


template <typename T>
void saveTensor(const tensor<T>& tensor, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return;
    }

    const auto* data = tensor.data();
    const auto dataSize = tensor.size() * sizeof(T);
    file.write(reinterpret_cast<const char*>(data), dataSize);
    file.close();
}

    
template <typename T>
tensor<T> loadTensor(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return tensor<T>();
    }

    std::vector<T> data;
    //read the data from the file,make sure that the size of the data is equal to the size of the tensor 
    //is smaller than than the maximum size allowed in memory
    if (file.fail()) {
        return tensor<T>();
    }
    //get the size from the file
    //seekg is used to move the file pointer to the beginning of the file 
    file.seekg(0, std::ios::end);
    const auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    //if the size of the file is greater than the maximum size allowed in memory 
    //return an empty tensor
    if (size >  MAX_TENSOR_SIZE * sizeof(T)) { 
        file.close();
        return tensor<T>();
    }   

    data.resize( size /  sizeof(T));
    file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(T));
    file.close();
    return tensor<T>(data);
}


