#include <stdio.h>                                                                                            
#include <stdlib.h>                                                                                           
                                                                                                              
#include <Python.h>                                                                                           
                                                                                                              
#define NPY_NO_DEPRECATED_API   NPY_1_7_API_VERSION                                                           
#include <numpy/arrayobject.h>                                                                                
                                                                                                              
#include "atsec/cuda_correlation.h"
                                                                                                              
static char module_docstring[] = "";                                
                                                                                                              
static PyObject*                                                                                              
correlation_correlate( PyObject *self, PyObject *args );                                               
static char correlation_correlate_docstring[] = "";                                                    
                                                                                                              
static PyMethodDef module_methods[] = {                                                                       
                                                                                                              
                                                                          
    {"correlate",                                    
      correlation_correlate, METH_VARARGS, correlation_correlate_docstring },                   
                                                                                                              
    {NULL, NULL, 0, NULL}                                                                                     
};                                                                                                            

PyMODINIT_FUNC init_correlation(void)
{
    PyObject *m = Py_InitModule3("_correlation", module_methods, module_docstring);
    if (m == NULL)
        return;

    /* Load `numpy` functionality. */
    import_array();
}


static PyObject *
correlation_correlate( PyObject *self, PyObject *args )
{

    PyObject *pyObj_res1, *pyObj_ims1;
    PyObject *pyObj_res2, *pyObj_ims2;

    // Parse input
    int err;
    err = PyArg_ParseTuple( args,
            "OOOO",
            &pyObj_res1, &pyObj_ims1, 
            &pyObj_res2, &pyObj_ims2 ); 
    if( !err )
        return NULL;

    // Parse python objects to numpy array objects
    PyArrayObject *pyArr_res1  =
        (PyArrayObject* )PyArray_FROM_OTF(
        pyObj_res1  ,
        NPY_FLOAT32,
        NPY_ARRAY_IN_ARRAY );
    
    PyArrayObject *pyArr_ims1  =
        (PyArrayObject* )PyArray_FROM_OTF(
        pyObj_ims1  ,
        NPY_FLOAT32,
        NPY_ARRAY_IN_ARRAY );
    
    PyArrayObject *pyArr_res2  =
        (PyArrayObject* )PyArray_FROM_OTF(
        pyObj_res2  ,
        NPY_FLOAT32,
        NPY_ARRAY_IN_ARRAY );
    
    PyArrayObject *pyArr_ims2  =
        (PyArrayObject* )PyArray_FROM_OTF(
        pyObj_ims2  ,
        NPY_FLOAT32,
        NPY_ARRAY_IN_ARRAY );
    
    // Get number of samples from dimension of signal1
    int nsamples    = (int  )PyArray_DIM (pyArr_res1 , 0 );

    // Parse numpy arrays to C-pointers
    float *re_s1  = (float *)PyArray_DATA(pyArr_res1);
    float *im_s1  = (float *)PyArray_DATA(pyArr_ims1);
    float *re_s2  = (float *)PyArray_DATA(pyArr_res2);
    float *im_s2  = (float *)PyArray_DATA(pyArr_ims2);
    
    int dims[1] = { 2*nsamples };
    PyArrayObject *pyArr_corr =(PyArrayObject *) PyArray_FromDims(1,dims,NPY_FLOAT);
    
    float *corr = (float *)PyArray_DATA(pyArr_corr);
  
    // Setup real transformation
    cudacorrelate_fft( re_s1, im_s1, re_s2, im_s2, nsamples, corr );
    
    return Py_BuildValue( "O", pyArr_corr);
}

