#ifndef READ_MAT_SPARSE_HPP
#define READ_MAT_SPARSE_HPP

#include <matio.h>
#include <iostream>
#include <string>

// Pass the matlab matrix at var, pass the column and row vector of the CSC by *reference* at jc_out and ir_out
static void getCSR(matvar_t *var, mat_uint32_t *&ir_out, mat_uint32_t *&jc_out);

// Function to get sparse matrix from a .mat file as a matlab variable
// param fileName: path to the .mat file
// param structName: name of the struct variable containing the sparse matrix
// param fieldName: name of the field in the struct that contains the sparse matrix
matvar_t* getSparseMatrix(const std::string &fileName,
                                const std::string &structName,
                                const std::string &fieldName);

#endif // READ_MAT_SPARSE_HPP