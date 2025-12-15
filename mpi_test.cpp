#include "helper_functions.h"
#include <mpi.h>
#include <omp.h>
#include <vector>
#include <chrono>
#include <map>
#include <iostream>
#include <algorithm>

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (argc < 2) {
        if (world_rank == 0) {
            std::cerr << "Usage: " << argv[0]
                      << " <mat-file> [structName] [fieldName] [K_sync]\n";
            std::cerr << "Default structName='Problem', fieldName='A', K_sync=1\n";
        }
        MPI_Finalize();
        return 1;
    }

    std::string fileName   = argv[1];
    std::string structName = (argc >= 3) ? argv[2] : "Problem";
    std::string fieldName  = (argc >= 4) ? argv[3] : "A";
    int K_sync             = (argc >= 5) ? std::max(1, std::atoi(argv[4])) : 1;

    size_t nrows = 0, ncols = 0, nnz = 0;

    // Rank 0 keeps full matrix
    std::vector<mat_uint32_t> row_ptr_full;
    std::vector<mat_uint32_t> col_idx_full;

    if (world_rank == 0) {
        matvar_t *sparseMatrix = getSparseMatrix(fileName, structName, fieldName);
        if (!sparseMatrix) MPI_Abort(MPI_COMM_WORLD, 1);

        mat_sparse_t *sparse = static_cast<mat_sparse_t*>(sparseMatrix->data);
        if (!sparse) {
            std::cerr << "Matrix is not sparse or data missing\n";
            Mat_VarFree(sparseMatrix);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        nrows = (sparseMatrix->rank >= 1) ? (size_t)sparseMatrix->dims[0] : 0;
        ncols = (sparseMatrix->rank >= 2) ? (size_t)sparseMatrix->dims[1] : 0;
        nnz   = (ncols > 0) ? (size_t)sparse->jc[ncols] : 0;

        std::cout << "Loaded " << fileName
                  << " struct=" << structName
                  << " field=" << fieldName
                  << " -> nrows=" << nrows
                  << " ncols=" << ncols
                  << " nnz="   << nnz
                  << " K_sync=" << K_sync
                  << "\n";

        row_ptr_full.resize(nrows + 1);
        col_idx_full.resize(nnz);

        // Keep your original handling (Matlab CSC arrays)
        std::copy(sparse->jc, sparse->jc + nrows + 1, row_ptr_full.data());
        std::copy(sparse->ir, sparse->ir + nnz,       col_idx_full.data());

        Mat_VarFree(sparseMatrix);
    }

    // Broadcast dims
    unsigned long long dims[3];
    if (world_rank == 0) {
        dims[0] = (unsigned long long)nrows;
        dims[1] = (unsigned long long)ncols;
        dims[2] = (unsigned long long)nnz;
    }
    MPI_Bcast(dims, 3, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    if (world_rank != 0) {
        nrows = (size_t)dims[0];
        ncols = (size_t)dims[1];
        nnz   = (size_t)dims[2];
    }

    // Full global labels on all ranks (same as your original)
    std::vector<int> A_l(nrows);
    for (size_t i = 0; i < nrows; ++i) A_l[i] = (int)(i + 1);

    // Partition rows
    size_t rows_per_proc = nrows / (size_t)world_size;
    size_t remainder     = nrows % (size_t)world_size;

    size_t row_start  = (size_t)world_rank * rows_per_proc + std::min((size_t)world_rank, remainder);
    size_t local_rows = rows_per_proc + ((size_t)world_rank < remainder ? 1 : 0);
    size_t row_end    = row_start + local_rows;

    std::vector<mat_uint32_t> row_ptr_local;
    std::vector<mat_uint32_t> col_idx_local;

    if (world_rank == 0) {
        for (int r = 0; r < world_size; ++r) {
            size_t r_row_start  = (size_t)r * rows_per_proc + std::min((size_t)r, remainder);
            size_t r_local_rows = rows_per_proc + ((size_t)r < remainder ? 1 : 0);
            size_t r_row_end    = r_row_start + r_local_rows;

            mat_uint32_t base = row_ptr_full[r_row_start];
            mat_uint32_t r_local_nnz = row_ptr_full[r_row_end] - base;

            if (r == 0) {
                row_ptr_local.resize(r_local_rows + 1);
                col_idx_local.resize((size_t)r_local_nnz);

                for (size_t i = 0; i <= r_local_rows; ++i)
                    row_ptr_local[i] = row_ptr_full[r_row_start + i] - base;

                std::copy(col_idx_full.begin() + base,
                          col_idx_full.begin() + (size_t)base + (size_t)r_local_nnz,
                          col_idx_local.begin());
            } else {
                unsigned long long meta[3];
                meta[0] = (unsigned long long)r_row_start;
                meta[1] = (unsigned long long)r_local_rows;
                meta[2] = (unsigned long long)r_local_nnz;
                MPI_Send(meta, 3, MPI_UNSIGNED_LONG_LONG, r, 0, MPI_COMM_WORLD);

                MPI_Send(row_ptr_full.data() + r_row_start,
                         (int)(r_local_rows + 1),
                         MPI_UNSIGNED,
                         r, 1, MPI_COMM_WORLD);

                MPI_Send(col_idx_full.data() + base,
                         (int)r_local_nnz,
                         MPI_UNSIGNED,
                         r, 2, MPI_COMM_WORLD);
            }
        }
    } else {
        unsigned long long meta[3];
        MPI_Recv(meta, 3, MPI_UNSIGNED_LONG_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        row_start  = (size_t)meta[0];
        local_rows = (size_t)meta[1];
        size_t local_nnz_recv = (size_t)meta[2];
        row_end = row_start + local_rows;

        std::vector<mat_uint32_t> row_ptr_segment(local_rows + 1);
        row_ptr_local.resize(local_rows + 1);
        col_idx_local.resize(local_nnz_recv);

        MPI_Recv(row_ptr_segment.data(),
                 (int)(local_rows + 1),
                 MPI_UNSIGNED,
                 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        mat_uint32_t base = row_ptr_segment[0];
        for (size_t i = 0; i <= local_rows; ++i)
            row_ptr_local[i] = row_ptr_segment[i] - base;

        MPI_Recv(col_idx_local.data(),
                 (int)local_nnz_recv,
                 MPI_UNSIGNED,
                 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    size_t local_nnz = (!row_ptr_local.empty()) ? (size_t)row_ptr_local[local_rows] : 0;

    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "Rank " << world_rank << "/" << world_size
              << " rows [" << row_start << "," << row_end << ")"
              << " local_rows=" << local_rows
              << " local_nnz=" << local_nnz
              << std::endl;

    // -----------------------------
    // Iterative label propagation with K-sync
    // -----------------------------
    bool global_changed = true;

    MPI_Barrier(MPI_COMM_WORLD);
    auto start_time = std::chrono::high_resolution_clock::now();

    long long outer_rounds = 0;

    while (global_changed) {
        bool local_changed_any = false;

        // Do K local iterations before syncing
        for (int t = 0; t < K_sync; ++t) {
            bool local_changed = false;

            #pragma omp parallel for schedule(static) reduction(||:local_changed)
            for (long long lr = 0; lr < (long long)local_rows; ++lr) {
                size_t local_row  = (size_t)lr;
                size_t global_row = row_start + local_row;

                int best = A_l[global_row];

                for (size_t idx = (size_t)row_ptr_local[local_row];
                            idx < (size_t)row_ptr_local[local_row + 1]; ++idx) {
                    size_t global_col = (size_t)col_idx_local[idx];
                    int v = A_l[global_col];
                    if (v < best) best = v;
                }

                if (best < A_l[global_row]) {
                    A_l[global_row] = best;
                    local_changed = true;
                }
            }

            local_changed_any = local_changed_any || local_changed;

            // optional micro early-exit: if nothing changed locally, no reason to keep looping t
            if (!local_changed) break;
        }

        // Global sync of full labels ONLY once per K batch
        MPI_Allreduce(MPI_IN_PLACE,
                      A_l.data(),
                      (int)nrows,
                      MPI_INT,
                      MPI_MIN,
                      MPI_COMM_WORLD);

        // Global convergence check once per K batch
        int loc = local_changed_any ? 1 : 0;
        int glob = 0;
        MPI_Allreduce(&loc, &glob, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        global_changed = (glob != 0);

        outer_rounds++;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    if (world_rank == 0) {
        std::cout << "Algorithm completed in " << elapsed.count()
                  << " seconds. outer_rounds=" << outer_rounds
                  << " K_sync=" << K_sync << "\n";

        std::map<int,int> label_count;
        for (size_t i = 0; i < nrows; ++i) label_count[A_l[i]]++;
        std::cout << "Different labels: " << label_count.size() << "\n";
    }

    MPI_Finalize();
    return 0;
}
