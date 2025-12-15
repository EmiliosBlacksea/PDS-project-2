#include "helper_functions.h"

// Pass the matlab matrix at var, pass the column and row vector of the CSC by *reference* at jc_out and ir_out
// The original matrix is symmetrical so the CSR can be formed from the stored CSC form by swapping jc and ir
static void getCSR(matvar_t *var, mat_uint32_t *&ir_out, mat_uint32_t *&jc_out)
{
    ir_out = nullptr;
    jc_out = nullptr;
    if (!var || !var->data)
        return;

    mat_sparse_t *sparse = static_cast<mat_sparse_t *>(var->data);
    // number of columns (assumes 2-D sparse matrix)
    mat_uint32_t ncols = (var->rank >= 2) ? static_cast<mat_uint32_t>(var->dims[1]) : 0;

    // determine number of nonzeros from sparse->jc
    mat_uint32_t nnz = sparse->jc[ncols];

    // copy jc (length = ncols + 1) into ir_out
    ir_out = new mat_uint32_t[ncols + 1];
    for (mat_uint32_t i = 0; i < ncols + 1; ++i)
    {
        ir_out[i] = sparse->jc[i];
    }

    // copy ir (length = nnz) into jc_out
    if (nnz > 0)
    {
        jc_out = new mat_uint32_t[nnz];
        for (mat_uint32_t i = 0; i < nnz; ++i)
        {
            jc_out[i] = sparse->ir[i];
        }
    }
    else
    {
        jc_out = nullptr;
    }

    // Note: caller is responsible for deleting the returned arrays
}



matvar_t* getSparseMatrix(const std::string &fileName,
                                const std::string &structName,
                                const std::string &fieldName){

    // Open the MAT file
    mat_t *matfp = Mat_Open(fileName.c_str(), MAT_ACC_RDONLY);
    if (matfp == nullptr) {
        std::cerr << "Error opening MAT file: " << fileName << std::endl;
        return nullptr; }


    // Read the struct variable
    // This is done this way because the database provided has this specific structure
    // If the .mat file only contains the matrix skip this part
    matvar_t *matStruct = Mat_VarRead(matfp, structName.c_str());
    if (matStruct == nullptr) {
        std::cerr << "Could not find struct: " << structName << std::endl;
        Mat_Close(matfp);
        return nullptr; }


    if (matStruct->class_type != MAT_C_STRUCT) {
        std::cerr << structName << " is not a MATLAB struct." << std::endl;
        Mat_VarFree(matStruct);
        Mat_Close(matfp);
        return nullptr; }

    // Read **only the field** (no deep copy needed)
    matvar_t *matField = Mat_VarGetStructFieldByName(matStruct, fieldName.c_str(), 0);
    
    if (matField->class_type != MAT_C_SPARSE) {
        std::cerr << "Field is not a sparse matrix." << std::endl;
        Mat_VarFree(matStruct);
        Mat_Close(matfp);
        return nullptr;
    }

    Mat_Close(matfp);

    return matField;  // caller is responsible for freeing this
                          }