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
#include <thread>

//lock guard
#include <mutex>
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
    protected:
    std::vector<uint32_t> last_scores = {};
    std::recursive_mutex py_forest_mutex;

    public:
    const std::vector<uint32_t>& get_last_scores() const {
        
    return last_scores;
    }

    //implement accessors for the forest 
    //constructor recieves properties of the forest 
    //actual forest is created when the first fit is called 
    //fit can be called with labeled data or without it.

    //constructor
    FastMatrixForestF32U32()
    {
        //guard mutex 
        std::lock_guard<std::recursive_mutex> lock(py_forest_mutex); 

        init_py();
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

        //initialize the list of the properties
        PyList_Append(py_forest_list,PyFloat_FromDouble(0.0)); 
        PyList_Append(py_forest_list,PyFloat_FromDouble(0.0));
        PyList_Append(py_forest_list,PyFloat_FromDouble(0.0));
        // finish the initialization of the list
        //register predict_proba
        PyType_Ready(&FastMatrixForestType); 

        
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

        std::vector<uint32_t> prediction_labels(labels.size());
        for(size_t i=0;i<labels.size();i++)
        {
            prediction_labels[i] = labels[i];
        }

        //fit the forest
        forest->fit(data,prediction_labels);
        //return the probabilities of the labels
        
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
        //store the results in the last_scores variable 
         last_scores = forest->predict(data);
         //debug print
         PySys_WriteStdout("[+] Predicted last scores %lu\n",last_scores.size());

        //return the labels
        return last_scores;
    }   
    std::vector<real_t> get_scores()
    {
        std::vector<real_t> scores(last_scores.size());
        for(size_t i=0;i<last_scores.size();i++)
        {
            scores[i] = 1.0-(last_scores[i]-i);
        }
        return scores;
        
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
        PyObject* ret = nullptr;
        
        try{
        //lock the mutex
        std::lock_guard<std::recursive_mutex> lock(py_forest_mutex); 

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
        //set return value:
        ret = py_prediction_probs;
        
        }catch(std::exception& e)
        {
            PyErr_SetString(PyExc_RuntimeError,e.what());
            return nullptr;
        }   

        return ret;

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
            //Debug output that the item was added : 
            PySys_WriteStdout("[+]Added item %zu\n",i);
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
    PyObject* predict_proba(PyObject* self , PyObject* args)    
    {
        PyObject* py_data = nullptr;
        //extract the arguments, if it's a single argument:
        if(args->ob_type == nullptr)
        {
            return nullptr;
        }
        else if(args->ob_type->tp_name == 0x0)
        {
            return nullptr;
        }
        else if(args->ob_type->tp_name == 0x0)
        {
            return nullptr;
        }
        std::string name = std::string(args->ob_type->tp_name);
        if(name.compare("tuple")==0)
        { 
            py_data = PyTuple_GetItem(args,0);
        }
        else if(name.compare("list")==0)
        {
            py_data = PyList_GetItem(args,0);
        }
        else
        {
            return nullptr;
        }
        //import numpy
        WrappedNumPy<real_t>::import_array();
        //get the data from numpy array
        provallo::matrix<real_t> data(1,1);
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
            data.resize(rows,cols);

            //get the data
            for (size_t i = 0; i < rows; i++)
            {
                for (size_t j = 0; j < cols; j++)
                {
                    data(i,j) = WrappedNumPy<real_t>::data(py_data)[i*cols+j];
                }
            }   
        }

        if(data.rows()==0)
        {
            PyErr_SetString(PyExc_RuntimeError,"Data is empty");
            return nullptr;
        }


        //predict the labels
        
        std::vector<real_t> prediction_probs = predict_proba_internal(data);
        //return the labels
        PyObject* py_prediction_probs = PyList_New(prediction_probs.size());
        for(size_t i=0;i<prediction_probs.size();i++)
        {
            PyList_SetItem(py_prediction_probs,i,PyFloat_FromDouble(prediction_probs[i]));
            //Debug output that the item was added : 
            PySys_WriteStdout("[+]Added item %zu\n",i);
        }
        return py_prediction_probs;
    } 
    //predict_proba_internal
    std::vector<real_t> predict_proba_internal(provallo::matrix<real_t> data) 
    {
        //predict the labels
        std::vector<real_t> prediction_probs;
        try{
            
            prediction_probs = forest->get_anomaly_score(data); 

        }catch(std::exception& e)
        {
            PyErr_SetString(PyExc_RuntimeError,e.what());
          
        } 
        return prediction_probs;


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
    // Initialize the Python object
    void init_py()
    {
        // Initialize the Python object
        PyObject* self_ = this;
        
        self_->ob_refcnt = 1;
        // Initialize the Python object
        
        self_->ob_type = &FastMatrixForestType;
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
        //check the args type and extract (list,tuple,dict,ndarray)
        if(args!=nullptr)
        {
            if(PyTuple_Check(args))
            {
                if(PyTuple_Size(args)==2)
                {
                    py_data = PyTuple_GetItem(args,0);
                    py_labels = PyTuple_GetItem(args,1);
                }
            }
            else if(PyList_Check(args))
            {
                if(PyList_Size(args)==2)
                {
                    py_data = PyList_GetItem(args,0);
                    py_labels = PyList_GetItem(args,1);
                }
            }
            else if(PyDict_Check(args))
            {
                py_data = PyDict_GetItemString(args,"data");
                py_labels = PyDict_GetItemString(args,"labels");
            }   
              
        }
        if(py_data==nullptr)
        {
            PyErr_SetString(PyExc_RuntimeError,"The data is not provided.");
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

