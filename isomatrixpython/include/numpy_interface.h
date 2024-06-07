#ifndef __NUMPY_INTERFACE_H__
#define __NUMPY_INTERFACE_H__

#include <Python.h>
#ifdef USE_NUMPY
#include <numpy/arrayobject.h>
#endif

template <typename T>
class WrappedNumPy {
public:
  static PyObject* import_array() {
    PyObject* mod = PyImport_ImportModule("numpy");
    // Handle import errors (same as before)
    return mod;
  }

  // Function wrappers using generic methods
  static int ndim(PyObject* obj) {
    // Check if the object has a `__array__` attribute
    PyObject* attr = PyObject_GetAttrString(obj, "__array__");
    if (attr == nullptr) {
      PyErr_SetString(PyExc_TypeError, "Object does not have __array__ attribute");
      return -1;
    }

    // Check if the attribute is callable (likely a NumPy array)
    if (!PyCallable_Check(attr)) {
      PyErr_SetString(PyExc_TypeError, "__array__ attribute is not callable");
      Py_DECREF(attr);
      return -1;
    }

    // Call the `__array__` method to potentially get a NumPy array
    PyObject* array = PyObject_CallObject(attr, nullptr);
    Py_DECREF(attr); // Release reference to attribute

    // Check if the returned object is a valid NumPy array (optional)
    // if ( !PyArray_Check(array) ) { ... handle non-array case ... }

    // Attempt to get the number of dimensions using generic methods
    int num_dims = -1;
    PyObject* shape_attr = PyObject_GetAttrString(array, "shape");
    if (shape_attr != nullptr) {
      if (PyTuple_Check(shape_attr)) {
        num_dims = PyTuple_Size(shape_attr);
      }
      Py_DECREF(shape_attr);
    }

    Py_DECREF(array); // Release reference obtained from __array__

    return num_dims;
  }

  // Similar approach for data access (considering type conversion)
  static T* data(PyObject* obj) {
    // Check if the object has a `__array__` attribute
    PyObject* attr = PyObject_GetAttrString(obj, "__array__");
    if (attr == nullptr) {
      PyErr_SetString(PyExc_TypeError, "Object does not have __array__ attribute");
      return nullptr;
    }

    // Check if the attribute is callable (likely a NumPy array)
    if (!PyCallable_Check(attr)) {
      PyErr_SetString(PyExc_TypeError, "__array__ attribute is not callable");
      Py_DECREF(attr);
      return nullptr;
    }

    // Call the `__array__` method to potentially get a NumPy array
    PyObject* array = PyObject_CallObject(attr, nullptr);
    Py_DECREF(attr); // Release reference to attribute

    // Check if the returned object is a valid NumPy array (optional)
    // if ( !PyArray_Check(array) ) { ... handle non-array case ... }

    // Attempt to get the data pointer using generic methods
    T* data_ptr = nullptr;
    PyObject* data_attr = PyObject_GetAttrString(array, "data");
    if (data_attr != nullptr) {
      if (PyCapsule_CheckExact(data_attr)) {
        data_ptr = static_cast<T*>(PyCapsule_GetPointer(data_attr, nullptr));
      }
      Py_DECREF(data_attr);
    }

    Py_DECREF(array); // Release reference obtained from __array__

    return data_ptr;    

  }
  //description of the array: shape, strides, data type, etc. 

  //get tuple of shape of the array 
    static PyObject* shape(PyObject* obj) {
        // Check if the object has a `__array__` attribute
        PyObject* attr = PyObject_GetAttrString(obj, "__array__");
        if (attr == nullptr) {
        PyErr_SetString(PyExc_TypeError, "Object does not have __array__ attribute");
        return nullptr;
        }
    
        // Check if the attribute is callable (likely a NumPy array)
        if (!PyCallable_Check(attr)) {
        PyErr_SetString(PyExc_TypeError, "__array__ attribute is not callable");
        Py_DECREF(attr);
        return nullptr;
        }
    
        // Call the `__array__` method to potentially get a NumPy array
        PyObject* array = PyObject_CallObject(attr, nullptr);
        Py_DECREF(attr); // Release reference to attribute
    
        // Check if the returned object is a valid NumPy array (optional)
        // if ( !PyArray_Check(array) ) { ... handle non-array case ... }
    
        // Attempt to get the shape tuple using generic methods
        PyObject* shape_attr = PyObject_GetAttrString(array, "shape");
        if (shape_attr == nullptr) {
        Py_DECREF(array); // Release reference obtained from __array__
        return nullptr;
        }
    
        // Increment reference count before returning
        Py_INCREF(shape_attr);
        Py_DECREF(array); // Release reference obtained from __array__
    
        return shape_attr;
    }   
    //get tuple of strides of the array
    static PyObject* strides(PyObject* obj) {
        // Check if the object has a `__array__` attribute
        PyObject* attr = PyObject_GetAttrString(obj, "__array__");
        if (attr == nullptr) {
        PyErr_SetString(PyExc_TypeError, "Object does not have __array__ attribute");
        return nullptr;
        }
    
        // Check if the attribute is callable (likely a NumPy array)
        if (!PyCallable_Check(attr)) {
        PyErr_SetString(PyExc_TypeError, "__array__ attribute is not callable");
        Py_DECREF(attr);
        return nullptr;
        }
    
        // Call the `__array__` method to potentially get a NumPy array
        PyObject* array = PyObject_CallObject(attr, nullptr);
        Py_DECREF(attr); // Release reference to attribute
    
        // Check if the returned object is a valid NumPy array (optional)
        // if ( !PyArray_Check(array) ) { ... handle non-array case ... }
    
        // Attempt to get the strides tuple using generic methods
        PyObject* strides_attr = PyObject_GetAttrString(array, "strides");
        if (strides_attr == nullptr) {
        Py_DECREF(array); // Release reference obtained from __array__
        return nullptr;
        }
    
        // Increment reference count before returning
        Py_INCREF(strides_attr);
        Py_DECREF(array); // Release reference obtained from __array__
    
        return strides_attr;
    }   
    //get data type of the array
    static PyObject* dtype(PyObject* obj) {
        // Check if the object has a `__array__` attribute
        PyObject* attr = PyObject_GetAttrString(obj, "__array__");
        if (attr == nullptr) {
        PyErr_SetString(PyExc_TypeError, "Object does not have __array__ attribute");
        return nullptr;
        }
    
        // Check if the attribute is callable (likely a NumPy array)
        if (!PyCallable_Check(attr)) {
        PyErr_SetString(PyExc_TypeError, "__array__ attribute is not callable");
        Py_DECREF(attr);
        return nullptr;
        }
    
        // Call the `__array__` method to potentially get a NumPy array
        PyObject* array = PyObject_CallObject(attr, nullptr);
        Py_DECREF(attr); // Release reference to attribute
    
        // Check if the returned object is a valid NumPy array (optional)
        // if ( !PyArray_Check(array) ) { ... handle non-array case ... }
    
        // Attempt to get the data type using generic methods
        PyObject* dtype_attr = PyObject_GetAttrString(array, "dtype");
        if (dtype_attr == nullptr) {
        Py_DECREF(array); // Release reference obtained from __array__
        return nullptr;
        }
    
        // Increment reference count before returning
        Py_INCREF(dtype_attr);
        Py_DECREF(array); // Release reference obtained from __array__
    
        return dtype_attr;
    }   
    //get the number of elements in the array
    static PyObject* size(PyObject* obj) {
        // Check if the object has a `__array__` attribute
        PyObject* attr = PyObject_GetAttrString(obj, "__array__");
        if (attr == nullptr) {
        PyErr_SetString(PyExc_TypeError, "Object does not have __array__ attribute");
        return nullptr;
        }
    
        // Check if the attribute is callable (likely a NumPy array)
        if (!PyCallable_Check(attr)) {
        PyErr_SetString(PyExc_TypeError, "__array__ attribute is not callable");
        Py_DECREF(attr);
        return nullptr;
        }
    
        // Call the `__array__` method to potentially get a NumPy array
        PyObject* array = PyObject_CallObject(attr, nullptr);
        Py_DECREF(attr); // Release reference to attribute
    
        // Check if the returned object is a valid NumPy array (optional)
        // if ( !PyArray_Check(array) ) { ... handle non-array case ... }
    
        // Attempt to get the number of elements using generic methods
        PyObject* size_attr = PyObject_GetAttrString(array, "size");
        if (size_attr == nullptr) {
        Py_DECREF(array); // Release reference obtained from __array__
        return nullptr;
        }
    
        // Increment reference count before returning
        Py_INCREF(size_attr);
        Py_DECREF(array); // Release reference obtained from __array__
    
        return size_attr;
    }   
    //get the item size of the array
    static PyObject* itemsize(PyObject* obj) {
        // Check if the object has a `__array__` attribute
        PyObject* attr = PyObject_GetAttrString(obj, "__array__");
        if (attr == nullptr) {
        PyErr_SetString(PyExc_TypeError, "Object does not have __array__ attribute");
        return nullptr;
        }
    
        // Check if the attribute is callable (likely a NumPy array)
        if (!PyCallable_Check(attr)) {
        PyErr_SetString(PyExc_TypeError, "__array__ attribute is not callable");
        Py_DECREF(attr);
        return nullptr;
        }
    
        // Call the `__array__` method to potentially get a NumPy array
        PyObject* array = PyObject_CallObject(attr, nullptr);
        Py_DECREF(attr); // Release reference to attribute
    
        // Check if the returned object is a valid NumPy array (optional)
        // if ( !PyArray_Check(array) ) { ... handle non-array case ... }
    
        // Attempt to get the item size using generic methods
        PyObject* itemsize_attr = PyObject_GetAttrString(array, "itemsize");
        if (itemsize_attr == nullptr) {
        Py_DECREF(array); // Release reference obtained from __array__
        return nullptr;
        }
    
        // Increment reference count before returning
        Py_INCREF(itemsize_attr);
        Py_DECREF(array); // Release reference obtained from __array__
    
        return itemsize_attr;
    }   
    //get the data pointer of the array
    static PyObject* data_ptr(PyObject* obj) {
        // Check if the object has a `__array__` attribute
        PyObject* attr = PyObject_GetAttrString(obj, "__array__");
        if (attr == nullptr) {
        PyErr_SetString(PyExc_TypeError, "Object does not have __array__ attribute");
        return nullptr;
        }
    
        // Check if the attribute is callable (likely a NumPy array)
        if (!PyCallable_Check(attr)) {
        PyErr_SetString(PyExc_TypeError, "__array__ attribute is not callable");
        Py_DECREF(attr);
        return nullptr;
        }
    
        // Call the `__array__` method to potentially get a NumPy array
        PyObject* array = PyObject_CallObject(attr, nullptr);
        Py_DECREF(attr); // Release reference to attribute
    
        // Check if the returned object is a valid NumPy array (optional)
        // if ( !PyArray_Check(array) ) { ... handle non-array case ... }
    
        // Attempt to get the data pointer using generic methods
        PyObject* data_attr = PyObject_GetAttrString(array, "data");
        if (data_attr == nullptr) {
        Py_DECREF(array); // Release reference obtained from __array__
        return nullptr;
        }
    
        // Increment reference count before returning
        Py_INCREF(data_attr);
        Py_DECREF(array); // Release reference obtained from __array__
    
        return data_attr;
    } 
    //get the flags of the array
    static PyObject* flags(PyObject* obj) {
        // Check if the object has a `__array__` attribute
        PyObject* attr = PyObject_GetAttrString(obj, "__array__");
        if (attr == nullptr) {
        PyErr_SetString(PyExc_TypeError, "Object does not have __array__ attribute");
        return nullptr;
        }
    
        // Check if the attribute is callable (likely a NumPy array)
        if (!PyCallable_Check(attr)) {
        PyErr_SetString(PyExc_TypeError, "__array__ attribute is not callable");
        Py_DECREF(attr);
        return nullptr;
        }
    
        // Call the `__array__` method to potentially get a NumPy array
        PyObject* array = PyObject_CallObject(attr, nullptr);
        Py_DECREF(attr); // Release reference to attribute
    
        // Check if the returned object is a valid NumPy array (optional)
        // if ( !PyArray_Check(array) ) { ... handle non-array case ... }
    
        // Attempt to get the flags using generic methods
        PyObject* flags_attr = PyObject_GetAttrString(array, "flags");
        if (flags_attr == nullptr) {
        Py_DECREF(array); // Release reference obtained from __array__
        return nullptr;
        }
    
        // Increment reference count before returning
        Py_INCREF(flags_attr);
        Py_DECREF(array); // Release reference obtained from __array__
    
        return flags_attr;
    } 

 };


#endif //__NUMPY_INTERFACE_H__