#ifndef __BIT_VECTOR_ATTRIBUTE_H__
#define __BIT_VECTOR_ATTRIBUTE_H__



#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <map>

 #include "matrix.h"
#include "optimizers.h"
//#include "dataset.h"
//#include "../util/csv_file.h"


//============================================================================== //
//--------------------------bit vector attribute-------------------------------- //
//             * this class is used to represent a bit vector attribute        * //
//--------------------------bit vector attribute-------------------------------- //
//============================================================================== // 
 
namespace provallo {

    #pragma pack(0)

    template <class T = uint8_t, size_t N = 8>
    class bit_type 
    {
        public :
        typedef T value_type;
        typedef T & reference;
        typedef const T & const_reference;
        typedef T * pointer;
        typedef const T * const_pointer;
        typedef T * iterator;

        static const size_t size = N;
        static const size_t npos = -1;
         

        private:
        value_type _bits; 
        public:
        bit_type() : _bits(0) {}
        bit_type(const bit_type &other) : _bits(other._bits) {}
        bit_type(bit_type &&other) : _bits(other._bits) {}
        bit_type(const T &other) : _bits(other) {}
        bit_type(T &&other) : _bits(other) {}
        


        bit_type & operator=(const bit_type &other) { 
            if( &other!=this) {
            _bits = other._bits;}
         return *this; 
         }
        bit_type & operator=(bit_type &&other) {
                if( &other!=this)
               { _bits = other._bits;} return *this; 
             }
        bit_type & operator=(const T &other) {
                if( &other!=this){
                _bits = other;} return *this;
             }
        bit_type & operator=(T &&other) { if(&other!=this){
         _bits = other;} return *this;
        }

        //arithmetic operators
        bit_type operator+(const bit_type &other) const { return bit_type(_bits + other._bits); } 
        bit_type operator-(const bit_type &other) const { return bit_type(_bits - other._bits); } 
        //division
        bit_type operator/(const bit_type &other) const { 
            //avoid floating point exception
            if(other._bits == 0 || _bits == 0) {
                return bit_type(0);
            } else {
                return bit_type(_bits / other._bits);
            }

        } 
        //modulus
        bit_type operator%(const bit_type &other) const { return bit_type(_bits % other._bits); } 
        //multiplication 
        bit_type operator*(const bit_type &other) const { return bit_type(_bits * other._bits); } 


        bool operator[](size_t i) const { return (_bits & (1 << i)) != 0; }
        bool operator[](size_t i) { return (_bits & (1 << i)) != 0; }

        bool operator[](int i) const { return (_bits & (1 << i)) != 0; }
        bool operator[](int i) { return (_bits & (1 << i)) != 0; }

        bool operator==(const bit_type &other) const { return _bits == other._bits; }
        bool operator!=(const bit_type &other) const { return _bits != other._bits; }
        bool operator<(const bit_type &other) const { return _bits < other._bits; }
        bool operator>(const bit_type &other) const { return _bits > other._bits; }
        bool operator<=(const bit_type &other) const { return _bits <= other._bits; }
        bool operator>=(const bit_type &other) const { return _bits >= other._bits; }
        bit_type operator&(const bit_type &other) const { return bit_type(_bits & other._bits); }
        bit_type operator|(const bit_type &other) const { return bit_type(_bits | other._bits); }
        bit_type operator^(const bit_type &other) const { return bit_type(_bits ^ other._bits); }
        bit_type operator~() const { return bit_type(~_bits); }
        bit_type & operator&=(const bit_type &other) { _bits &= other._bits; return *this; }
        bit_type & operator|=(const bit_type &other) { _bits |= other._bits; return *this; }
        bit_type & operator^=(const bit_type &other) { _bits ^= other._bits; return *this; }
        bit_type & operator<<=(const bit_type &other) { _bits <<= other._bits; return *this; }
        bit_type & operator>>=(const bit_type &other) { _bits >>= other._bits; return *this; }
        bit_type operator<<(const bit_type &other) const { return bit_type(_bits << other._bits); }
        bit_type operator>>(const bit_type &other) const { return bit_type(_bits >> other._bits); }
   //     bit_type operator<<(size_t i) const { return bit_type(_bits << i); }
   //     bit_type operator>>(size_t i) const { return bit_type(_bits >> i); }
        bit_type & operator<<=(size_t i) { _bits <<= i; return *this; }

        bit_type & operator>>=(size_t i) { _bits >>= i; return *this; }
        bit_type operator&(const T &other) const { return bit_type(_bits & other); }
        bit_type operator|(const T &other) const { return bit_type(_bits | other); }
        bit_type operator^(const T &other) const { return bit_type(_bits ^ other); }
        //bit_type operator~() const { return bit_type(~_bits); }
        bit_type & operator&=(const T &other) { _bits &= other; return *this; }
        bit_type & operator|=(const T &other) { _bits |= other; return *this; }
        bit_type & operator^=(const T &other) { _bits ^= other; return *this; }
        bit_type & operator<<=(const T &other) { _bits <<= other; return *this; }
        bit_type & operator>>=(const T &other) { _bits >>= other; return *this; }
        bit_type operator<<(const T &other) const { return bit_type(_bits << other); }
        bit_type operator>>(const T &other) const { return bit_type(_bits >> other); }
       //bit_type operator<<(size_t i) const { return bit_type(_bits << i); }
        //bit_type operator>>(size_t i) const { return bit_type(_bits >> i); }

        //bit_type & operator<<=(size_t i) { _bits <<= i; return *this; }
        //bit_type & operator>>=(size_t i) { _bits >>= i; return *this; }
        //operator == for T:
        bool operator==(const T &other) const { return _bits == other; } 
        bool operator!=(const T &other) const { return _bits != other; }

        bit_type & set() { _bits = ~0; return *this; }
        bit_type & reset() { _bits = 0; return *this; }
        bit_type & flip() { _bits = ~_bits; return *this; }
        bit_type & set(size_t i) { _bits |= (1 << i); return *this; }
        bit_type & reset(size_t i) { _bits &= ~(1 << i); return *this; }
        bit_type & flip(size_t i) { _bits ^= (1 << i); return *this; }
        bit_type & set(size_t i, bool v) { if (v) set(i); else reset(i); return *this; }
        bit_type & reset(size_t i, bool v) { if (v) reset(i); else set(i); return *this; }
        bit_type & flip(size_t i, bool v) { if (v) flip(i); return *this; }
        bit_type & set(size_t i, const T &v) { if (v) set(i); else reset(i); return *this; }
        bit_type & reset(size_t i, const T &v) { if (v) reset(i); else set(i); return *this; }
        bit_type & flip(size_t i, const T &v) { if (v) flip(i); return *this; }
        bit_type & set(size_t i, const bit_type &v) { if (v[i]) set(i); else reset(i); return *this; }
        bit_type & reset(size_t i, const bit_type &v) { if (v[i]) reset(i); else set(i); return *this; }
        bit_type & flip(size_t i, const bit_type &v) { if (v[i]) flip(i); return *this; }
        bit_type & set(size_t i, const bit_type &v, bool b) { if (v[i]) set(i, b); else reset(i, b); return *this; }
        bit_type & reset(size_t i, const bit_type &v, bool b) { if (v[i]) reset(i, b); else set(i, b); return *this; }
        bit_type & flip(size_t i, const bit_type &v, bool b) { if (v[i]) flip(i, b); return *this; }
        bit_type & set(size_t i, const bit_type &v, const T &b) { if (v[i]) set(i, b); else reset(i, b); return *this; }
        bit_type & reset(size_t i, const bit_type &v, const T &b) { if (v[i]) reset(i, b); else set(i, b); return *this; }
        bit_type & flip(size_t i, const bit_type &v, const T &b) { if (v[i]) flip(i, b); return *this; }
        bit_type & set(size_t i, const bit_type &v, const bit_type &b) { if (v[i]) set(i, b); else reset(i, b); return *this; }
        bit_type & reset(size_t i, const bit_type &v, const bit_type &b) { if (v[i]) reset(i, b); else set(i, b); return *this; }
        //bool operator[](size_t i) const { return (_bits >> i) & 1; } 

        friend bool operator==(const T &a, const bit_type &b) { return a == b._bits; }  
        friend bool operator!=(const T &a, const bit_type &b) { return a != b._bits; }
        friend bool operator<(const T &a, const bit_type &b) { return a < b._bits; }
        friend bool operator>(const T &a, const bit_type &b) { return a > b._bits; }
        friend bool operator<=(const T &a, const bit_type &b) { return a <= b._bits; }
        friend bool operator>=(const T &a, const bit_type &b) { return a >= b._bits; }

        friend bool operator==(const bit_type &a, const T &b) { return a._bits == b; }
        friend bool operator!=(const bit_type &a, const T &b) { return a._bits != b; }
        friend bool operator<(const bit_type &a, const T &b) { return a._bits < b; }
        friend bool operator>(const bit_type &a, const T &b) { return a._bits > b; }

        friend bool operator<=(const bit_type &a, const T &b) { return a._bits <= b; }
        friend bool operator>=(const bit_type &a, const T &b) { return a._bits >= b; }
        
        T & value() { return _bits; }
        const T & value() const { return _bits; }
        operator T() const { return _bits; }
        operator T&() { return _bits; }
        operator const T&() const { return _bits; }
        operator T*() { return &_bits; }
        operator const T*() const { return &_bits; }
        operator T&() const { return _bits; }
        operator const T&() { return _bits; }
        std::string to_string() const { return std::to_string(_bits); } 
        T sum() const {  
            T sum = 0;
            for (size_t i = 0; i <  N; ++i) {
                if ((*this)[i]) {
                    sum |= (T)1 << i;
                }
            }
            return sum;
        }
        
     };

    typedef std::vector<bit_type<uint8_t,8>> u_bit_vector;
    typedef std::vector<bit_type<uint16_t,16>> u_bit_vector16;
    typedef std::vector<bit_type<uint32_t,32>> u_bit_vector32;
    typedef std::vector<bit_type<uint64_t,64>> u_bit_vector64;
    typedef std::vector<bit_type<uint8_t,8>> u_bit_vector8;
    typedef std::vector<bit_type<int8_t,8>> s_bit_vector8;
    typedef std::vector<bit_type<int16_t,16>> s_bit_vector16;
    typedef std::vector<bit_type<int32_t,32>> s_bit_vector32;
    typedef std::vector<bit_type<int64_t,64>> s_bit_vector64;
    typedef std::vector<bit_type<int8_t,8>> s_bit_vector;

    typedef std::vector<bit_type<float,32>> f_bit_vector32;
    typedef std::vector<bit_type<double,64>> f_bit_vector64;
    typedef std::vector<bit_type<float,32>> f_bit_vector;
    typedef std::vector<bit_type<bool,1>> b_bit_vector;

    template <class T,size_t N>
    std::ostream & operator<<(std::ostream &out, const bit_type<T,N> &b)
    {
        for (size_t i = 0; i < N; i++)
         out << b[i];
       
        return out;
    }   


    template <class T,size_t N>
    std::istream & operator>>(std::istream &in, bit_type<T,N> &b)
    {
        for (size_t i = 0; i < N; i++)
        {
            char c;
            in >> c;
            b[i] = c == '1';
        }
        return in;
    }   
    //to string
    template <class T,size_t N>
    std::string to_string(const bit_type<T,N> &b)
    {
        std::stringstream ss;
        ss << b;
        return ss.str();
    }   
    //n
    template <class T,size_t N> 
    std::string to_string(const bit_type<T,N> &b, size_t n)
    {
        std::stringstream ss;
        for (size_t i = 0; i < n; i++)
            ss << b[i];
        return ss.str();
    }   
    //separator     
    template <class T,size_t N>
    std::string to_string(const bit_type<T,N> &b, const std::string &sep)
    {
        std::stringstream ss;
        for (size_t i = 0; i < N; i++)
        {
            if (i > 0)
                ss << sep;
            ss << b[i];
        }
        return ss.str();
    } 
    template <class T,size_t N>
    bit_type<T,N>  unique_set_to_bit_type( const std::vector<T> &v)
    {
        bit_type<T,N> b;
        for (size_t i = 0; i < v.size(); i++)
            b.set(v[i]);
        return b;
    } 
    template <class T>
    std::vector<T>  
    unique_subset(const std::vector<T>& unique) 
    {

        std::vector<T> subset;
        for (size_t i = 0; i < unique.size(); i++)
            {
                //if not in subset, add it
                if (std::find(subset.begin(), subset.end(), unique[i]) == subset.end())
                    subset.push_back(unique[i]);
                    
            }
        return subset;
    }


    template <class T>
    std::vector<T>  
    unique_subset(  std::vector<T>& unique) 
    {
        //sort
        std::sort(unique.begin(), unique.end());
        //remove duplicates
        std::vector<T> subset;
        for (size_t i = 0; i < unique.size(); i++)
            {
                //if not in subset, add it
                if (std::find(subset.begin(), subset.end(), unique[i]) == subset.end())
                    subset.push_back(unique[i]);
                    
            }
        return subset;
    }
    //specialize unique subset for real_t
    template <>
    std::vector<real_t>
    unique_subset(  std::vector<real_t>& unique) 
    {
        //sort
        std::sort(unique.begin(), unique.end());
        real_t min=unique[0];
        real_t max=unique[unique.size()-1];
        
        //remove duplicates
        std::vector<real_t> subset;
        for (size_t i = 0; i < unique.size(); i++)
            {
                //if not in subset, add it
                if (std::find(subset.begin(), subset.end(), unique[i]) == subset.end())
                    subset.push_back(unique[i]);
                    
            } 
        size_t steps = subset.size();
        real_t step = (max-min)/steps;
        for (size_t i = 0; i < subset.size(); i++)
        {
                subset[i] = min + i*step;
        }
        return subset;
    }   


    //---------------------------------------------------------------------------------//   
    //bitwise operators:
    //---------------------------------------------------------------------------------//

    template <class T,size_t N> 
    bit_type<T,N> operator&(const bit_type<T,N> &a, const bit_type<T,N> &b)
    {
        return a & b;
    }
    template <class T,size_t N> 
    bit_type<T,N> operator|(const bit_type<T,N> &a, const bit_type<T,N> &b)
    {
        return a | b;
    } 
    template <class T,size_t N> 
    bit_type<T,N> operator^(const bit_type<T,N> &a, const bit_type<T,N> &b)
    {
        return a ^ b;
    } 
    template <class T,size_t N> 
    bit_type<T,N> operator~(const bit_type<T,N> &a)
    {
        return ~a;
    } 
    template <class T,size_t N> 
    bit_type<T,N> operator<<(const bit_type<T,N> &a, const bit_type<T,N> &b)
    {
        return a << b;
    }
    template <class T,size_t N>
    bit_type<T,N> operator>>(const bit_type<T,N> &a, const bit_type<T,N> &b)
    {
        return a >> b;
    }
    template <class T,size_t N>
    bit_type<T,N> operator<<(const bit_type<T,N> &a, size_t i)
    {
        return a << i;
    } 
    template <class T,size_t N> 
    bit_type<T,N> operator>>(const bit_type<T,N> &a, size_t i)
    {
        return a >> i;
    } 

    //---------------------------------------------------------------------------------//
    //comparison operators:
    //---------------------------------------------------------------------------------//


    template <class T,size_t N>
    bool operator==(const bit_type<T,N> &a, const bit_type<T,N> &b)
    {
        return a == b;
    }
    template <class T,size_t N>
    bool operator!=(const bit_type<T,N> &a, const bit_type<T,N> &b)
    {
        return a != b;
    }
    template <class T,size_t N>
    bool operator<(const bit_type<T,N> &a, const bit_type<T,N> &b)
    {
        return a < b;
    }
    template <class T,size_t N>
    bool operator>(const bit_type<T,N> &a, const bit_type<T,N> &b)
    {
        return a > b;
    }
    template <class T,size_t N>
    bool operator<=(const bit_type<T,N> &a, const bit_type<T,N> &b)
    {
        return a <= b;
    }
    template <class T,size_t N>
    bool operator>=(const bit_type<T,N> &a, const bit_type<T,N> &b)
    {
        return a >= b;
    }
    //---------------------------------------------------------------------------------//

    
    
    
    //dna sequence:
    typedef bit_type<uint8_t,3> dna_base; 
    typedef std::vector<dna_base> dna_sequence; 
    typedef matrix<dna_base> dna_matrix; 
    //definitions:
    static const dna_base dnaA(0); //0,0,0
    static const dna_base dnaC(1) ;  //0,0,1 ;
    static const dna_base dnaG(2); //0,1,0
    static const dna_base dnaT(3); //0,1,1
    static const dna_base dnaN(4); //1,0,0

    

    
    static const std::vector<dna_base> dna_bases =  {dnaA,dnaC,dnaG,dnaT,dnaN}; 

    //oligomer:
    typedef std::vector<dna_base> oligomer;
    //oligomer sequence:
    typedef std::vector<oligomer> oligomer_sequence; 
    //oligomer matrix:
    typedef matrix<oligomer> oligomer_matrix;

    //oligoneucleotide:
    typedef std::vector<dna_base> oligoneucleotide;
    //oligoneucleotide sequence:
    typedef std::vector<oligoneucleotide> oligoneucleotide_sequence; 
    //oligoneucleotide matrix:
    typedef matrix<oligoneucleotide> oligoneucleotide_matrix; 

    //protein base:
    typedef bit_type<uint8_t,8> protein_base;
    //protein constant 
    //protein base makes up the amino acid sequence. 
    //basic components of each protein sequence are called amino acids. 
    //amino acids are encoded by 8 bits.
    typedef std::vector<protein_base> protein_sequence; 
    //protein matrix:
    typedef matrix<protein_base> protein_matrix;

    //definitions:
    static const protein_base pA(0); //0,0,0,0,0,0,0,0
    static const protein_base pC(1) ;  //0,0,0,0,0,0,0,1 ;
    static const protein_base pD(2); //0,0,0,0,0,0,1,0
    static const protein_base pE(3); //0,0,0,0,0,0,1,1
    static const protein_base pF(4); //0,0,0,0,0,1,0,0
    static const protein_base pG(5); //0,0,0,0,0,1,0,1
    static const protein_base pH(6); //0,0,0,0,0,1,1,0
    static const protein_base pI(7); //0,0,0,0,0,1,1,1
    static const protein_base pK(8); //0,0,0,0,1,0,0,0
    static const protein_base pL(9); //0,0,0,0,1,0,0,1
    static const protein_base pM(10); //0,0,0,0,1,0,1,0
    static const protein_base pN(11); //0,0,0,0,1,0,1,1
    static const protein_base pP(12); //0,0,0,0,1,1,0,0
    static const protein_base pQ(13); //0,0,0,0,1,1,0,1
    static const protein_base pR(14); //0,0,0,0,1,1,1,0
    static const protein_base pS(15); //0,0,0,0,1,1,1,1
    static const protein_base pV(16); //0,0,0,1,0,0,0,0
    static const protein_base pW(17); //0,0,0,1,0,0,0,1
    static const protein_base pY(18); //0,0,0,1,0,0,1,0
    
    static const protein_base pSTOP(19); //0,0,0,1,0,0,1,1 

    static const std::vector<protein_base> protein_bases = {pA,pC,pD,pE,pF,pG,pH,pI,pK,pL,pM,pN,pP,pQ,pR,pS,pV,pW,pY,pSTOP};
    
    //cell base:
    typedef bit_type<uint8_t,3> cell_base;
    //cell constant 
    //cell type (T-cell, B-cell, etc) is encoded by 3 bits.

    typedef std::vector<cell_base> cell_sequence; 
    //cell matrix:
    typedef matrix<cell_base> cell_matrix;

    //definitions:
    static const cell_base cT(0); //0,0,0
    static const cell_base cB(1); //0,0,1
    static const cell_base cN(2); //0,1,0
    static const cell_base cM(3); //0,1,1
    static const cell_base cG(4); //1,0,0
    static const cell_base cC(5); //1,0,1
    static const cell_base cR(6); //1,1,0
    static const cell_base cP(7); //1,1,1
    static const std::vector<cell_base> cell_bases = {cT,cB,cN,cM,cG,cC,cR,cP};

    //exosome base: 
    typedef bit_type<uint8_t,3> exosome_base;

    typedef std::vector<exosome_base> exosome_sequence;
    //exosome matrix:
    typedef matrix<exosome_base> exosome_matrix;

    
    //definitions:
    static const exosome_base eA(0); //0,0,0
    static const exosome_base eB(1); //0,0,1
    static const exosome_base eC(2); //0,1,0
    static const exosome_base eD(3); //0,1,1
    static const exosome_base eE(4); //1,0,0
    static const exosome_base eF(5); //1,0,1
    static const exosome_base eG(6); //1,1,0
    static const exosome_base eH(7); //1,1,1
    static const std::vector<exosome_base> exosome_bases = {eA,eB,eC,eD,eE,eF,eG,eH}; 


    //mrna base:
    typedef bit_type<uint8_t,3> mrna_base; 
    typedef std::vector<mrna_base> mrna_sequence; 
    typedef matrix<mrna_base> mrna_matrix; 
    //definitions:
    static const mrna_base mA(0); //0,0,0
    static const mrna_base mC(1); //0,0,1
    static const mrna_base mG(2); //0,1,0
    static const mrna_base mU(3); //0,1,1
    static const mrna_base mN(4); //1,0,0
    static const mrna_base mS(5); //1,0,1
    static const mrna_base mT(6); //1,1,0
    static const mrna_base mW(7); //1,1,1
    static const std::vector<mrna_base> mrna_bases = {mA,mC,mG,mU,mN,mS,mT,mW}; 


    #pragma pack()

}

#endif 