#ifndef __PYTHON_INTERFACE_H__
#define __PYTHON_INTERFACE_H__

#include <Python.h>


#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <memory>
#include <algorithm>
#include <iterator>
#include <type_traits>
#include <functional>
#include <utility>
#include <tuple>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
//stdint :
#include <stdint.h>
#include <ctype.h>
#include <list>
#include "matrix.h"
#include "fast_matrix_forest.h"
#include "numpy_interface.h" 

extern PyTypeObject FastMatrixForestType ;
  class FastMatrixForestF32U32 : public PyObject {
    
    
    private:
    provallo::super_tree<real_t,uint32_t>* forest=nullptr;
    PyObject* py_forest = nullptr; 
    PyObject* py_forest_dict = nullptr;
    PyObject* py_forest_list = nullptr;
    public:

    //implement accessors for the forest 
    //constructor recieves properties of the forest 
    //actual forest is created when the first fit is called 
    //fit can be called with labeled data or without it.

    //constructor
    FastMatrixForestF32U32()
    {
        InitPy();
        //initialize the dictionary of the properties

        if(py_forest_dict==nullptr)
        {
            py_forest_dict = PyDict_New();
        }
        
        if(py_forest_list==nullptr)
        {
            py_forest_list = PyList_New(0);
        }

        //initialize the dictionary of the properties
        PyDict_SetItemString(py_forest_dict,"eta",PyFloat_FromDouble(0.0)); 
        PyDict_SetItemString(py_forest_dict,"momentum",PyFloat_FromDouble(0.0));
        PyDict_SetItemString(py_forest_dict,"divergence",PyFloat_FromDouble(0.0));
        // finish the initialization of the dictionary 
        
    }
    //fit method:
    //fit method is called with the data and the labels and returns probabilities of the labels 
    std::vector<real_t> fit_internal(const provallo::matrix<real_t>& data, const std::vector<uint32_t>& labels)
    {
        //create the forest if it does not exist
        if(forest!=nullptr)
        {
            delete forest;
            
        }
        forest = new provallo::super_tree<real_t,uint32_t>(data,labels,data.rows(),data.cols()  );


        //fit the forest
        forest->fit(data,labels);
        //return the probabilities of the labels
        auto prediction_labels = forest->predict(data);
        std::vector<real_t> prediction_probs(prediction_labels.size());
        for(size_t i=0;i<prediction_labels.size();i++)
        {
            prediction_probs[i] = 1.0-(prediction_labels[i]-labels[i]);
        }   
        return prediction_probs;
    } 
    std::vector<uint32_t> predict_internal(const provallo::matrix<real_t>& data)
    {
        //create the forest if it does not exist
        if(forest==nullptr)
        {
            throw std::runtime_error("The forest is not fitted yet.");
        }
        //predict the labels
        return forest->predict(data);
    }   
    //destructor
    ~FastMatrixForestF32U32()
    {
        if(forest!=nullptr)
        {
            delete forest;
        }
        if(py_forest_dict!=nullptr)
        {
            Py_DECREF(py_forest_dict);
        }   
        if(py_forest_list!=nullptr)
        {
            Py_DECREF(py_forest_list);
        }   

    }
    //fit method with numpy arrays recieved from Python:
    PyObject* fit(PyObject* py_data, PyObject* py_labels)
    {
        //import numpy
        try{
        WrappedNumPy<real_t>::import_array();
        //get the data and labels from numpy arrays
        provallo::matrix<real_t> data;
        std::vector<uint32_t> labels;
        //get the data
        if(py_data!=nullptr)
        {
            //get the dimensions of the data :
            PyObject* py_shape = WrappedNumPy<real_t>::shape(py_data); 
            //get the number of dimensions
            int ndim = PyList_Size(py_shape);
            //get the shape of the data
            std::vector<size_t> shape(ndim);
            for(int i=0;i<ndim;i++)
            {
                shape[i] = PyLong_AsLong(PyList_GetItem(py_shape,i));
            }

            size_t rows = shape[0];
            size_t cols = shape[1];
            provallo::matrix<real_t> data(rows,cols);

            //get the data
            for (size_t i = 0; i < rows; i++)
            {
                for (size_t j = 0; j < cols; j++)
                {
                    data(i,j) = WrappedNumPy<real_t>::data(py_data)[i*cols+j];
                }
            }   

        }
        //get the labels
        if(py_labels!=nullptr)
        {
            //get the labels
            //get the shape/size of the labels vector:
            PyObject* py_size = WrappedNumPy<uint32_t>::size(py_labels); 
            size_t size = PyLong_AsLong(py_size);
            //get the labels
            for(size_t i=0;i<size;i++)
            {
                labels.push_back(WrappedNumPy<uint32_t>::data(py_labels)[i]);
            }   

        }
        //fit the forest
        std::vector<real_t> prediction_probs = fit_internal(data,labels);
        //return the probabilities of the labels
        PyObject* py_prediction_probs = PyList_New(prediction_probs.size());
        for(size_t i=0;i<prediction_probs.size();i++)
        {
            PyList_SetItem(py_prediction_probs,i,PyFloat_FromDouble(prediction_probs[i]));
        }

        return py_prediction_probs;
        }catch(std::exception& e)
        {
            PyErr_SetString(PyExc_RuntimeError,e.what());
            return nullptr;
        }   

    }   
    //predict method with numpy arrays recieved from Python:
    PyObject* predict(PyObject* py_data)
    {
        //import numpy
        WrappedNumPy<real_t>::import_array();
        //get the data from numpy array
        provallo::matrix<real_t> data;
        //get the data
        if(py_data!=nullptr)
        {
            //get the data
            //get the dimensions of the data :
            PyObject* py_shape = WrappedNumPy<real_t>::shape(py_data);
            //get the number of dimensions
            int ndim = PyList_Size(py_shape);
            //get the shape of the data
            std::vector<size_t> shape(ndim);
            for(int i=0;i<ndim;i++)
            {
                shape[i] = PyLong_AsLong(PyList_GetItem(py_shape,i));
            }

            size_t rows = shape[0];
            size_t cols = shape[1];
            provallo::matrix<real_t> data(rows,cols);

            //get the data
            for (size_t i = 0; i < rows; i++)
            {
                for (size_t j = 0; j < cols; j++)
                {
                    data(i,j) = WrappedNumPy<real_t>::data(py_data)[i*cols+j];
                }
            }   
            

        }
        //predict the labels
        std::vector<uint32_t> prediction_labels = predict_internal(data);
        //return the labels
        PyObject* py_prediction_labels = PyList_New(prediction_labels.size());
        for(size_t i=0;i<prediction_labels.size();i++)
        {
            PyList_SetItem(py_prediction_labels,i,PyLong_FromLong(prediction_labels[i]));
        }
        return py_prediction_labels;
    }   
    //get the super tree properties:
    PyObject* get_properties()
    {
        //create the dictionary of the properties
        if(py_forest_dict==nullptr)
        {
            py_forest_dict = PyDict_New();
        }
        //add the properties to the dictionary
        if(forest!=nullptr)
        {
            //add the properties to the dictionary
            PyDict_SetItemString(py_forest_dict,"eta",PyFloat_FromDouble(forest->get_eta()));
            //PyDict_SetItemString(py_forest_dict,"gamma",PyFloat_FromDouble(forest->get_gamma()));
            PyDict_SetItemString(py_forest_dict,"momentum",PyFloat_FromDouble(forest->get_momentum())); 
            //divergence
            PyDict_SetItemString(py_forest_dict,"divergence",PyFloat_FromDouble(forest->divergent_stability())); 
            //add the properties of the super tree
        }   
        return py_forest_dict;
    }
    //get the super tree properties:
    PyObject* get_properties_list()
    {
        //create the dictionary of the properties
        if(py_forest_list==nullptr)
        {
            py_forest_list = PyList_New(0);
        }
        //add the properties to the dictionary
        if(forest!=nullptr)
        {
            //add the properties to the dictionary
            PyList_Append(py_forest_list,PyFloat_FromDouble(forest->get_eta()));
            //PyDict_SetItemString(py_forest_dict,"gamma",PyFloat_FromDouble(forest->get_gamma()));
            PyList_Append(py_forest_list,PyFloat_FromDouble(forest->get_momentum())); 
            //divergence
            PyList_Append(py_forest_list,PyFloat_FromDouble(forest->divergent_stability())); 
            //add the properties of the super tree
        }   
        return py_forest_list;
    }   
    //PyObject initialization:
    void InitPy()
    {
        //initialize the Python object:
        PyObject* self = this;
        
        self->ob_refcnt = 1;
        //initialize the Python object:
        
        self->ob_type =  &FastMatrixForestType; 
        
    }
    static void dealloc(FastMatrixForestF32U32* self)
    {
        if( (PyObject*)self!=nullptr)
        {
            //delete the object:
            delete self;
            //free the memory:
            Py_TYPE(self)->tp_free((PyObject*)self);

        }
    }   
    static PyObject* fit(FastMatrixForestF32U32* self, PyObject* args)
    {
        PyObject* py_data = nullptr;
        PyObject* py_labels = nullptr;
        if(!PyArg_ParseTuple(args,"OO",&py_data,&py_labels))
        {
            return nullptr;
        }
        return self->fit(py_data,py_labels);
    }   
    static PyObject* predict(FastMatrixForestF32U32* self, PyObject* args)
    {
        PyObject* py_data = nullptr;
        if(!PyArg_ParseTuple(args,"O",&py_data))
        {
            return nullptr;
        }
        return self->predict(py_data);
    }   
    static PyObject* get_properties(FastMatrixForestF32U32* self, PyObject* args)
    {
        return self->get_properties();
    }   
    static PyObject* get_properties_list(FastMatrixForestF32U32* self, PyObject* args)
    {
        return self->get_properties_list();
    }   
    
    
   };//end of the class FastMatrixForestF32U32


#endif //__PYTHON_INTERFACE_H__

