//include all the necessary headers for python interface 

// Path: src/fast_matrix_forest.cpp 
//include all the necessary headers for the fast matrix forest
//use FastMatrixForest class to create a forest of fast matrix trees from Python 
#include "../include/python_interface.h"
#include "../include/fast_matrix_forest.h" 
#include "../include/autoencoder.h"
#include "../include/lstm.h"
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <memory>
#include <algorithm>
#include <iterator> 


typedef FastMatrixForestF32U32 FastMatrixForest;

   //Python interface for the FastMatrixForestF32U32 class:
    //create the Python module:
    extern "C"  PyObject* FastMatrixForestF32U32_new(PyTypeObject* type, PyObject* args, PyObject* kwds)
    {
        //get the data and the labels:
        PyObject* py_data = nullptr;
        PyObject* py_labels = nullptr; 
        if (!args)
        {
            return Py_None;
        }
        //parse the arguments,check if the arguments are a tuple,ndarray or a list: 
        std::string name = std::string(args->ob_type->tp_name);
        if (name.compare("tuple") == 0)
        {
            py_data = PyTuple_GetItem(args,0);
            py_labels = PyTuple_GetItem(args,1);

        }
        else if (name.compare("list") == 0)
        {
            
            py_data = PyList_GetItem(args,0);
            py_labels = PyList_GetItem(args,1);

        }
        else if (name.compare("ndarray") == 0)
        {
            
            py_data = args;
            py_labels = PyTuple_GetItem(args,1);

        }

        //check if the arguments are a tuple,ndarray or a list:
        if(py_data == nullptr || py_labels == nullptr)
        {
            throw std::invalid_argument("Invalid arguments, please provide a tuple,ndarray or a list as input"); 
        }
        //create the object:
        FastMatrixForestF32U32* self = new FastMatrixForestF32U32();

        //fit the object:
        if(self!=nullptr)
            self->fit(py_data,py_labels); 
        return (PyObject*)self;
        
    }   
    //destructor:
    extern "C" void FastMatrixForestF32U32_dealloc(FastMatrixForestF32U32* self)
    {
        Py_TYPE(self)->tp_free((PyObject*)self);
    }   
    

    //fit method:
    extern "C" PyObject* FastMatrixForestF32U32_fit(FastMatrixForestF32U32* self, PyObject* args)
    {
        PyObject* py_data = nullptr; 
        PyObject* py_labels = nullptr;
        //check if args are a tuple,ndarray or a list and extract the data and the labels: 
        if(args->ob_type->tp_name=="tuple")
        {
            py_data = PyTuple_GetItem(args,0);
            py_labels = PyTuple_GetItem(args,1);
        }
        else if(args->ob_type->tp_name=="list")
        {
            py_data = PyList_GetItem(args,0);
            py_labels = PyList_GetItem(args,1);
        }
        else if(args->ob_type->tp_name=="ndarray")
        {
            py_data = args;
            py_labels = PyTuple_GetItem(args,1);
        }
        else
        {
            return nullptr;
        }

        return self->fit(py_data,py_labels);
    }   
    extern "C" PyObject* fit(FastMatrixForestF32U32* self, PyObject* args)
    {
        return FastMatrixForestF32U32_fit(self,args);
    }   
    //predict method:
    extern "C" PyObject* FastMatrixForestF32U32_predict(FastMatrixForestF32U32* self, PyObject* args)
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
        else if(name.compare("ndarray")==0)
        {
            py_data = args;
        }
        else
        {
            return nullptr;
        }
        //check if args are a tuple,ndarray or a list and extract the data:
        
        return self->predict(py_data);
    }
    //get the properties of the super tree:
    extern "C" PyObject* FastMatrixForestF32U32_get_properties(FastMatrixForestF32U32* self, PyObject* args)
    {
        //ignore the arguments:
        UNDEF_REFERENCE(args);
        UNDEF_REFERENCE2(args);

        return self->get_properties();
    }
    //get the properties of the super tree:
    extern "C" PyObject* FastMatrixForestF32U32_get_properties_list(FastMatrixForestF32U32* self, PyObject* args)
    {
        return self->get_properties_list();
    }
    //define the methods of the class:
    PyMethodDef FastMatrixForest_methods[] = {
        {"fit",(PyCFunction)FastMatrixForestF32U32_fit,METH_VARARGS,"Fit the forest with the data and the labels."},
        {"predict",(PyCFunction)FastMatrixForestF32U32_predict,METH_VARARGS,"Predict the labels of the data."},
        {"get_properties",(PyCFunction)FastMatrixForestF32U32_get_properties,METH_VARARGS,"Get the properties of the super tree."},
        {"get_properties_list",(PyCFunction)FastMatrixForestF32U32_get_properties_list,METH_VARARGS,"Get the properties of the super tree."},
        //new() and delete() methods:
        {"new",(PyCFunction)FastMatrixForestF32U32_new,METH_VARARGS,"Create a new FastMatrixForest object."},
        {"delete",(PyCFunction)FastMatrixForestF32U32_dealloc,METH_VARARGS,"Delete the FastMatrixForest object."},
        //add interfaces :
        {"FastMatrixForest_create_python_fast_matrix_forest",(PyCFunction)FastMatrixForestF32U32_new,METH_VARARGS,"Create a new FastMatrixForest object."},
        {"FastMatrixForest_delete_python_fast_matrix_forest",(PyCFunction)FastMatrixForestF32U32_dealloc,METH_VARARGS,"Delete the FastMatrixForest object."}, 
        {"FastMatrixForest_fit",(PyCFunction)FastMatrixForestF32U32_fit,METH_VARARGS,"Fit the forest with the data and the labels."},
        {"FastMatrixForest_predict",(PyCFunction)FastMatrixForestF32U32_predict,METH_VARARGS,"Predict the labels of the data."},
        {"FastMatrixForest_get_properties",(PyCFunction)FastMatrixForestF32U32_get_properties,METH_VARARGS,"Get the properties of the super tree."},
        {"FastMatrixForest_get_properties_list",(PyCFunction)FastMatrixForestF32U32_get_properties_list,METH_VARARGS,"Get the properties of the super tree."},
        
        
        {nullptr,nullptr,0,nullptr}
    };
    //define the class:
    PyTypeObject FastMatrixForestType = {
        PyVarObject_HEAD_INIT(nullptr,0)
        "FastMatrixForest",
        sizeof(FastMatrixForestF32U32),
        0,
        (destructor)FastMatrixForestF32U32_dealloc
    };
    //define the module:
    PyModuleDef FastMatrixForestF32U32Module = {
        PyModuleDef_HEAD_INIT,
        "FastMatrixForest",
        "FastMatrixForest module",
        -1,
        nullptr
    };
    //define the module init function:
    extern "C" PyMODINIT_FUNC PyInit_FastMatrixForest()
    {
        PyObject* m;
        if(PyType_Ready(&FastMatrixForestType)<0)
        {
            return nullptr;
        }
        m = PyModule_Create(&FastMatrixForestF32U32Module);
        if(m==nullptr)
        {
            return nullptr;
        }
        Py_INCREF(&FastMatrixForestType);
        PyModule_AddObject(m,"FastMatrixForest",(PyObject*)&FastMatrixForestType);
        return m;
    }   
    //new/delete functions:
    extern "C" PyObject* FastMatrixForest_new(PyTypeObject* type, PyObject* args, PyObject* kwds)
    {
        FastMatrixForest* self =  new FastMatrixForest();
        self->ob_type = type;        
        return (PyObject*)self;
    }   
    extern "C" void FastMatrixForest_delete(FastMatrixForest* self)
    {
        if( (PyObject*)self!=nullptr)
        {
            //delete the object:
            delete self;
            //free the memory:
            Py_TYPE(self)->tp_free((PyObject*)self);

        }
    }   
    //fit method:
    extern "C" PyObject* FastMatrixForest_fit(FastMatrixForest* self, PyObject* args)
    {
        PyObject* py_data = nullptr;
        PyObject* py_labels = nullptr;
        if(!PyArg_ParseTuple(args,"OO",&py_data,&py_labels))
        {
            return nullptr;
        }
        return self->fit(py_data,py_labels);
    }   
    //predict method:
    extern "C" PyObject* FastMatrixForest_predict(FastMatrixForest* self, PyObject* args)
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
        else if(strncmp(args->ob_type->tp_name,"tuple",5)==0)
        {
            py_data = PyTuple_GetItem(args,0);
        }
        else if(strncmp(args->ob_type->tp_name,"list",4)==0)
        {
            py_data = PyList_GetItem(args,0);
        }
        else if(strncmp(args->ob_type->tp_name,"ndarray",7)==0)
        {
            py_data = args;
        }
        else
        {
            return nullptr;
        }   
        if(py_data!=nullptr)
        {
            return self->predict(py_data);
        }
        else
        {
            return nullptr;
        }   
    }   
    //get the properties of the super tree:
    extern "C" PyObject* FastMatrixForest_get_properties(FastMatrixForest* self, PyObject* args)
    {
        return self->get_properties();
    }   
    //get the properties of the super tree: 
    extern "C" PyObject* FastMatrixForest_get_properties_list(FastMatrixForest* self, PyObject* args)
    {
        return self->get_properties_list();
    }   
#if 0 
extern "C" PyObject* FastMatrixForestF32U32_get_properties_list(FastMatrixForestF32U32* self, PyObject* args)   
{
    return FastMatrixForest_get_properties_list((FastMatrixForest*)self,args);  
}

extern "C" PyObject* FastMatrixForestF32U32_new(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
    return FastMatrixForest_new((PyTypeObject*)&FastMatrixForestType,args,kwds);

}
extern "C" void FastMatrixForestF32U32_dealloc(FastMatrixForestF32U32* self)
{
    FastMatrixForest_delete(self);  
}

#endif
const std::string to_string(PyObject* obj)
{
    std::stringstream ss;
    ss<<"<"<<obj->ob_type->tp_name<<">";
    ss<<"("<<obj<<")";
     return ss.str();
}

extern "C" PyObject* FastMatrixForest_create_python_fast_matrix_forest(PyObject* self, PyObject* args , PyObject* kwds =nullptr, PyObject* extra_args = nullptr, PyObject* extra_kwds = nullptr )
{
    PyObject* py_data = nullptr;
    PyObject* py_labels = nullptr;
    //validate the content of the argument pointer:
    if(args == nullptr)
    {
        return nullptr;
    }   
    //validate the type of the argument:

    if(args->ob_type == nullptr || args->ob_type->tp_name == 0x0) 
    {
         return nullptr;
    }
    if(args->ob_type->tp_name ==  nullptr)
    {
         return nullptr;
    }
   
    std::string name = std::string(args->ob_type->tp_name);

    if(name.compare("tuple")==0)
    {
        py_data = PyTuple_GetItem(args,0);
        py_labels = PyTuple_GetItem(args,1);

    }
    else if(name.compare("list")==0)
    {
        py_data = PyList_GetItem(args,0);
        py_labels = PyList_GetItem(args,1);
    }
    else if(name.compare("ndarray")==0)
    {
        py_data = args;
        py_labels = PyTuple_GetItem(args,1);
    }

    
    FastMatrixForestF32U32* self_ = new FastMatrixForestF32U32();
    self_->ob_type = &FastMatrixForestType;
    //fit the object:

    //check if it's a tuple,ndarray or a list and extract the data and the labels: 
    if(py_data == nullptr || py_labels == nullptr)
    {
        return nullptr;
    }

    self_->fit(py_data,py_labels);
    
    
    return (PyObject*)self_;


}   


extern "C" PyObject* FastMatrixForest_delete_python_fast_matrix_forest(PyObject* self, PyObject* args)
{
     //get the pointer to the object:
    FastMatrixForestF32U32* self_ = (FastMatrixForestF32U32*)PyCapsule_GetPointer(self,"FastMatrixForest");
    //delete the object:
    FastMatrixForestF32U32_dealloc(self_);
    //return None:
    Py_RETURN_NONE;


}   

//FastMatrixForest_get_scores:
extern "C" PyObject* FastMatrixForest_get_scores(FastMatrixForest* self, PyObject* args)
{
    //make sure that the arguments are valid:
    if(args == nullptr)
    {
      
      PyErr_SetString(PyExc_RuntimeError,"Warning: no arguments passed to get_scores"); 
        
    }

    //check for null pointer:
    if(self == nullptr)
    {
        return nullptr;
    }

    //get the scores:
    auto scores = self->get_scores(); 

    //make sure that the scores are valid:
    if(scores.size() == 0)
    {
        PyErr_SetString(PyExc_RuntimeError,"Warning: no scores returned from the get_scores");
    }
    else if(scores.size() == 1)
    {
        PyErr_SetString(PyExc_RuntimeError,"Warning: only one score returned from the get_scores");
    }
    
    if (args != nullptr)
    {
        //Py_DECREF(args);

        //assign the scores passed as argument:
        if(PyTuple_Check(args))
        {
            //get the number of arguments passed:   
            Py_ssize_t size = PyTuple_Size(args);

            if(size != scores.size())
            {
                PyErr_SetString(PyExc_RuntimeError,"Warning: the number of scores passed in the get_scores does not match the number of scores returned");
            }

            for(size_t i=0;i<scores.size();i++)
            {
                PyTuple_SetItem(args,i,Py_BuildValue("f",scores[i]));
            }
        }
        else if(PyList_Check(args))
        {
            //get the number of arguments passed:   
            Py_ssize_t size = PyList_Size(args);    

            if(size != scores.size())
            {
                PyErr_SetString(PyExc_RuntimeError,"Warning: the number of scores passed in the get_scores does not match the number of scores returned");
            }

            for(size_t i=0;i<scores.size();i++)
            {
                PyList_SetItem(args,i,Py_BuildValue("f",scores[i]));
            }
        }
        else
        {
            PyErr_SetString(PyExc_RuntimeError,"Warning: the argument passed in the get_scores is not a tuple or a list");
        }


    }

    //return npy array with the scores for each data point:
    
    return Py_BuildValue("n",scores.data(), static_cast<Py_ssize_t>(scores.size()));

} 

    
        
    //end of the Python interface for the FastMatrixForestF32U32 class

// Path: src/fast_matrix_tree.cpp
//get the accuracy of the super_tree<float32,uint32_t> on the data:
std::atomic_uint64_t provallo::tag_hyperplane::hplane_count(0); 
