#ifndef AMIGO_CSR_DISTRIBUTE_H
#define AMIGO_CSR_DISTRIBUTE_H

#include <mpi.h>

#include "node_owners.h"
#include "ordering_utils.h"

namespace amigo {

class MatDistribute {
 public:
  template <typename T>
  class MatDistributeContext {
   public:
    TacsScalar *ext_A;          // Pointer to the data accumulated on this proc
    MPI_Request *ext_requests;  // Requests for sending info

    MPI_Request *in_requests;  // Requests for recving data
    TacsScalar *in_A;
  };

  std::shared_ptr<NodeOwners> row_owners;
  std::shared_ptr<NodeOwners> col_owners;

  // Data destined for other processes
  // ---------------------------------
  int num_ext_procs;  // Number of processors that will be sent data
  int num_ext_rows;   // Total number of rows that are send data
  int *ext_procs;     // External proc numbers
  int *ext_count;     // Number of rows sent to each proc
  int *ext_row_ptr;   // Pointer from proc into the ext_rows array
  int *ext_rows;      // Row indices
  int *ext_rowp;      // Pointer into the rows
  int *ext_cols;      // Global column indices

  // Data received from other processes
  // ----------------------------------
  int num_in_procs;  // Number of processors that give contributions
  int num_in_rows;   // Total number of rows given by other processors
  int *in_procs;     // Processor numbers that give info (num_in_procs)
  int *in_count;     // Count of rows from other processors (num_in_procs)
  int *in_row_ptr;   // Offset from proc index to location in rows
  int *in_rows;      // Row numbers for each row (num_in_rows)
  int *in_rowp;      // Pointer into the column numbers (num_in_rows)
  int *in_cols;      // Global column indices

  MatDistribute(MPI_Comm comm, std::shared_ptr<NodeOwners> row_owners,
                std::shared_ptr<NodeOwners> col_owners, CSRMat<T> *csr)
      : comm(owners->get_mpi_comm()),
        row_owners(row_owners),
        col_owners(col_owners) {
    int mpi_size, mpi_rank;
    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Comm_size(comm, &mpi_size);

    // Find the number of rows and columns
    int nrows = ;
    int ncols = ;

    // Get the owner range for the number of variables owned by
    // each node
    const int *row_ranges;
    row_owners->get_ranges(&row_ranges);

    // Get the non-zero pattern of the input matrix
    int lower = row_ranges[mpi_rank];
    int upper = row_ranges[mpi_rank + 1];

    // Count up the number of equations that must be sent to other
    // processors. Get only the variables associated with those ones.
    num_ext_rows = 0;
    for (int k = 0; k < nrows; k++) {
      if (ext_vars[k] < lower || ext_vars[k] >= upper) {
        num_ext_rows++;
      }
    }

    ext_rows = new int[num_ext_rows];
    num_ext_rows = 0;
    for (int k = 0; k < nrows; k++) {
      if (ext_vars[k] < lower || ext_vars[k] >= upper) {
        ext_rows[num_ext_rows] = ext_vars[k];
        num_ext_rows++;
      }
    }

    // Create the off-process CSR data structure that will be sent to
    // other processes. First, calculate ext_rowp
    ext_rowp = new int[num_ext_rows + 1];
    std::fill(ext_rowp, ext_rowp + (num_ext_rows + 1), 0);

    for (int i = 0; i < num_nodes; i++) {
      if (ext_vars[i] < lower || ext_vars[i] >= upper) {
        int *item = TacsSearchArray(ext_vars[i], num_ext_rows, ext_rows);
        int index = item - ext_rows;
        ext_rowp[index + 1] = rowp[i + 1] - rowp[i];
      }
    }

    for (int i = 0; i < num_ext_rows; i++) {
      ext_rowp[i + 1] += ext_rowp[i];
    }

    // Next, calculate ext_cols. Find only the external rows and
    // convert them to the global numbering scheme
    ext_cols = new int[ext_rowp[num_ext_rows]];

    for (int i = 0; i < numNodes; i++) {
      if (ext_vars[i] < lower || ext_vars[i] >= upper) {
        int *item = TacsSearchArray(ext_vars[i], num_ext_rows, ext_rows);
        int index = item - ext_rows;

        for (int j = ext_rowp[index], jj = rowp[i]; j < ext_rowp[index + 1];
             j++, jj++) {
          ext_cols[j] = ext_vars[cols[jj]];
        }

        int size = ext_rowp[index + 1] - ext_rowp[index];
        if (size != TacsUniqueSort(size, &ext_cols[ext_rowp[index]])) {
          fprintf(stderr, "[%d] TACSMatDistribute error: Array is not unique\n",
                  mpiRank);
        }
      }
    }

    // Match the intervals of the external variables to be sent to other
    // processes
    int *ext_ptr = new int[mpiSize + 1];
    OrderingUtils::match_intervals(mpiSize, row_range, num_ext_rows, ext_rows,
                                   ext_ptr);

    // Count up the processors that will be sending information
    num_ext_procs = 0;
    for (int k = 0; k < mpiSize; k++) {
      int num_rows = ext_ptr[k + 1] - ext_ptr[k];
      if (num_rows > 0) {
        num_ext_procs++;
      }
    }

    // Find the external processors
    ext_procs = new int[num_ext_procs];
    ext_count = new int[num_ext_procs];
    ext_row_ptr = new int[num_ext_procs + 1];
    ext_row_ptr[0] = 0;
    for (int k = 0, count = 0; k < mpiSize; k++) {
      int num_rows = ext_ptr[k + 1] - ext_ptr[k];
      if (num_rows > 0) {
        ext_procs[count] = k;
        ext_count[count] = num_rows;
        ext_row_ptr[count + 1] = ext_row_ptr[count] + num_rows;
        count++;
      }
    }

    // Adjust the pointer array so that it is an array of counts
    for (int k = 0, offset = 0; k < mpiSize; k++) {
      ext_ptr[k] = ext_ptr[k + 1] - offset;
      offset = ext_ptr[k + 1];
    }

    // Allocate space to store the number of in-coming entries
    int *in_full = new int[mpiSize];
    MPI_Alltoall(ext_ptr, 1, MPI_INT, in_full, 1, MPI_INT, comm);
    delete[] ext_ptr;

    // Count up the number of processors contributing entries to
    // this procesor
    num_in_procs = 0;
    for (int k = 0; k < mpiSize; k++) {
      if (in_full[k] > 0) {
        num_in_procs++;
      }
    }

    // Allocate space to store which processors are sendin
    in_procs = new int[num_in_procs];
    in_count = new int[num_in_procs];
    in_row_ptr = new int[num_in_procs + 1];
    in_row_ptr[0] = 0;
    for (int k = 0, count = 0; k < mpiSize; k++) {
      if (in_full[k] > 0) {
        in_procs[count] = k;
        in_count[count] = in_full[k];
        in_row_ptr[count + 1] = in_row_ptr[count] + in_count[count];
        count++;
      }
    }

    // Prepare to receive the equation numbers from the other processes
    in_row_ptr = new int[num_in_procs + 1];
    in_row_ptr[0] = 0;
    for (int k = 0; k < num_in_procs; k++) {
      in_row_ptr[k + 1] = in_row_ptr[k] + in_count[k];
    }

    // Allocate space for the integers
    num_in_rows = in_row_ptr[num_in_procs];
    in_rows = new int[num_in_rows];

    // Allocate space for the statuses
    in_requests = new MPI_Request[num_in_procs];
    ext_requests = new MPI_Request[num_ext_procs];

    // Post the recvs
    for (int k = 0, offset = 0; k < num_in_procs; k++) {
      int count = in_count[k];
      int source = in_procs[k];
      int tag = 1;
      MPI_Irecv(&in_rows[offset], count, MPI_INT, source, tag, comm,
                &in_requests[k]);
      offset += count;
    }

    // Post the sends
    for (int k = 0, offset = 0; k < num_ext_procs; k++) {
      int count = ext_count[k];
      int dest = ext_procs[k];
      int tag = 1;
      MPI_Isend(&ext_rows[offset], count, MPI_INT, dest, tag, comm,
                &ext_requests[k]);
      offset += count;
    }

    // Wait until everything completes
    if (num_ext_procs > 0) {
      MPI_Waitall(num_ext_procs, ext_requests, MPI_STATUSES_IGNORE);
    }
    if (num_in_procs > 0) {
      MPI_Waitall(num_in_procs, in_requests, MPI_STATUSES_IGNORE);
    }

    // Prepare to pass information from ext_rowp to in_rowp
    int *ext_row_var_count = new int[num_ext_rows];
    for (int k = 0; k < num_ext_rows; k++) {
      ext_row_var_count[k] = ext_rowp[k + 1] - ext_rowp[k];
    }

    in_rowp = new int[num_in_rows + 1];
    in_rowp[0] = 0;

    // Post the recvs
    for (int k = 0, offset = 1; k < num_in_procs; k++) {
      int count = in_count[k];
      int source = in_procs[k];
      int tag = 2;
      MPI_Irecv(&in_rowp[offset], count, MPI_INT, source, tag, comm,
                &in_requests[k]);
      offset += count;
    }

    // Post the sends
    for (int k = 0, offset = 0; k < num_ext_procs; k++) {
      int count = ext_count[k];
      int dest = ext_procs[k];
      int tag = 2;
      MPI_Isend(&ext_row_var_count[offset], count, MPI_INT, dest, tag, comm,
                &ext_requests[k]);
      offset += count;
    }

    if (num_ext_procs > 0) {
      MPI_Waitall(num_ext_procs, ext_requests, MPI_STATUSES_IGNORE);
    }
    if (num_in_procs > 0) {
      MPI_Waitall(num_in_procs, in_requests, MPI_STATUSES_IGNORE);
    }

    // Convert the counts to a pointer offset
    for (int k = 0; k < num_in_rows; k++) {
      in_rowp[k + 1] += in_rowp[k];
    }

    delete[] ext_row_var_count;

    // Pass ext_cols to in_cols
    in_cols = new int[in_rowp[num_in_rows]];

    // Post the recvs
    for (int k = 0, offset = 0, buff_offset = 0; k < num_in_procs; k++) {
      int count = in_rowp[offset + in_count[k]] - in_rowp[offset];
      int source = in_procs[k];
      int tag = 3;
      MPI_Irecv(&in_cols[buff_offset], count, MPI_INT, source, tag, comm,
                &in_requests[k]);
      offset += in_count[k];
      buff_offset += count;
    }

    // Post the sends
    for (int k = 0, offset = 0, buff_offset = 0; k < num_ext_procs; k++) {
      int count = ext_rowp[offset + ext_count[k]] - ext_rowp[offset];
      int dest = ext_procs[k];
      int tag = 3;
      MPI_Isend(&ext_cols[buff_offset], count, MPI_INT, dest, tag, comm,
                &ext_requests[k]);
      offset += ext_count[k];
      buff_offset += count;
    }

    if (num_ext_procs > 0) {
      MPI_Waitall(num_ext_procs, ext_requests, MPI_STATUSES_IGNORE);
    }
    if (num_in_procs > 0) {
      MPI_Waitall(num_in_procs, in_requests, MPI_STATUSES_IGNORE);
    }

    // estimate the non-zero entries per row
    int nz_per_row = 10;
    if (numNodes > 0) {
      nz_per_row = (int)(rowp[numNodes] / numNodes + 1);
    }

    int max_col_vars_size = num_ext_rows;
    for (int i = 0; i < num_in_rows; i++) {
      for (int j = in_rowp[i]; j < in_rowp[i + 1]; j++) {
        if (in_cols[j] < ownerRange[mpiRank] ||
            in_cols[j] >= ownerRange[mpiRank + 1]) {
          max_col_vars_size++;
        }
      }
    }

    int col_vars_size = 0;
    int *temp_col_vars = new int[max_col_vars_size];

    // Get contributions from ext_rows
    for (int i = 0; i < num_ext_rows; i++) {
      temp_col_vars[col_vars_size] = ext_rows[i];
      col_vars_size++;
    }

    for (int i = 0; i < num_in_rows; i++) {
      for (int j = in_rowp[i]; j < in_rowp[i + 1]; j++) {
        if (in_cols[j] < ownerRange[mpiRank] ||
            in_cols[j] >= ownerRange[mpiRank + 1]) {
          temp_col_vars[col_vars_size] = in_cols[j];
          col_vars_size++;
        }
      }
    }

    // Uniquely sort the array
    col_vars_size = TacsUniqueSort(col_vars_size, temp_col_vars);

    int *col_vars = new int[col_vars_size];
    memcpy(col_vars, temp_col_vars, col_vars_size * sizeof(int));
    delete[] temp_col_vars;

    TACSBVecIndices *col_indices =
        new TACSBVecIndices(&col_vars, col_vars_size);
    *colMap = new TACSBVecDistribute(row_map, col_indices);
    col_map_size = col_indices->getIndices(&col_map_vars);

    // Assemble everything into on and off-diagonal parts
    int *Arowp, *Acols;  // The diagonal entries
    int np, *Browp, *Bcols;

    computeLocalCSR(numNodes, ext_vars, rowp, cols, ownerRange[mpiRank],
                    ownerRange[mpiRank + 1], nz_per_row, &Arowp, &Acols, &np,
                    &Browp, &Bcols);

    // Allocate the local/external matrices
    int n = ownerRange[mpiRank + 1] - ownerRange[mpiRank];
    *Aloc = new BCSRMat(comm, thread_info, bsize, n, n, &Arowp, &Acols);
    *Bext = new BCSRMat(comm, thread_info, bsize, n - np, col_vars_size, &Browp,
                        &Bcols);

    // Allocate space for in-coming matrix elements
    ext_A = new TacsScalar[bsize * bsize * ext_rowp[num_ext_rows]];
    in_A = new TacsScalar[bsize * bsize * in_rowp[num_in_rows]];
    zeroEntries();
  }

  ~TACSMatDistribute() {
    row_map->decref();
    delete[] ext_procs;
    delete[] ext_count;
    delete[] ext_row_ptr;
    delete[] ext_rows;
    delete[] ext_rowp;
    delete[] ext_cols;
    delete[] ext_A;
    delete[] ext_requests;

    delete[] in_procs;
    delete[] in_count;
    delete[] in_row_ptr;
    delete[] in_rows;
    delete[] in_rowp;
    delete[] in_cols;
    delete[] in_A;
    delete[] in_requests;
  }

  /*!
    Find the non-zero pattern for the A = [B, E; F, C] and Bext
    matrices such that:

    [ B, E ][ x ]
    [ F, C ][ y ] + [ Bext ][ y_ext ] = 0
  */
  void computeLocalCSR(int numNodes, const int *ext_vars, const int *rowp,
                       const int *cols, int lower, int upper, int nz_per_row,
                       int **_Arowp, int **_Acols, int *_np, int **_Browp,
                       int **_Bcols) {
    int mpiRank, mpiSize;
    MPI_Comm_rank(comm, &mpiRank);
    MPI_Comm_size(comm, &mpiSize);

    // For each row, find the non-zero pattern
    int A_max_row_size = 2 * nz_per_row;
    int *A_row_vars = new int[A_max_row_size];

    int B_max_row_size = 2 * nz_per_row;
    int *B_row_vars = new int[B_max_row_size];

    int A_rows = upper - lower;
    int *A_rowp = new int[A_rows + 1];
    int *B_rowp = new int[A_rows + 1];

    for (int i = 0; i < A_rows + 1; i++) {
      A_rowp[i] = 0;
      B_rowp[i] = 0;
    }

    // Set up a temporary array to keep track of the variables in ext_vars
    int *var_map = new int[A_rows];
    for (int i = 0; i < A_rows; i++) {
      var_map[i] = -1;
    }

    for (int i = 0; i < numNodes; i++) {
      int var = ext_vars[i];
      if (var >= lower && var < upper) {
        var_map[var - lower] = i;
      }
    }

    // Size the A_rowp/B_rowp arrays
    for (int i = 0; i < A_rows; i++) {  // For each variable
      int A_row_size = 0;
      int B_row_size = 0;

      int ei = var_map[i];  // Index into the external variable map
      int var = lower + i;

      if (ei >= 0) {
        // Add variables in this range to the row as well
        int start = rowp[ei];
        int end = rowp[ei + 1];

        // Add rowp[ei], to A_row_vars, B_row_vars
        if (A_row_size + end - start > A_max_row_size) {
          A_max_row_size = A_max_row_size + end - start;
          TacsExtendArray(&A_row_vars, A_row_size, A_max_row_size);
        }

        if (B_row_size + end - start > B_max_row_size) {
          B_max_row_size = B_max_row_size + end - start;
          TacsExtendArray(&B_row_vars, B_row_size, B_max_row_size);
        }

        for (int j = start; j < end; j++) {
          int col_var = ext_vars[cols[j]];
          if (col_var >= lower && col_var < upper) {
            A_row_vars[A_row_size] = col_var;
            A_row_size++;
          } else {
            B_row_vars[B_row_size] = col_var;
            B_row_size++;
          }
        }
      }

      // Merge the off-processor contributions to the rows of A/B
      for (int k = 0; k < num_in_procs; k++) {
        // Try to find the variable in the k-th input - these are sorted
        // locally between in_rows[in_row_ptr[k]:in_row_ptr[k+1]]
        int count = in_row_ptr[k + 1] - in_row_ptr[k];
        int *item = TacsSearchArray(var, count, &in_rows[in_row_ptr[k]]);

        if (item) {
          int row = item - &in_rows[in_row_ptr[k]];
          row += in_row_ptr[k];

          // Add variables in this range to the row as well
          int start = in_rowp[row];
          int end = in_rowp[row + 1];

          if (A_row_size + end - start > A_max_row_size) {
            A_max_row_size = A_max_row_size + end - start;
            TacsExtendArray(&A_row_vars, A_row_size, A_max_row_size);
          }

          if (B_row_size + end - start > B_max_row_size) {
            B_max_row_size = B_max_row_size + end - start;
            TacsExtendArray(&B_row_vars, B_row_size, B_max_row_size);
          }

          for (int j = start; j < end; j++) {
            int col_var = in_cols[j];
            if (col_var >= lower && col_var < upper) {
              A_row_vars[A_row_size] = col_var;
              A_row_size++;
            } else {
              B_row_vars[B_row_size] = col_var;
              B_row_size++;
            }
          }
        }
      }

      // Sort the entries and remove duplicates
      A_row_size = TacsUniqueSort(A_row_size, A_row_vars);
      A_rowp[var - lower + 1] = A_row_size;

      B_row_size = TacsUniqueSort(B_row_size, B_row_vars);
      B_rowp[var - lower + 1] = B_row_size;
    }

    // Now, set up A_rowp/B_rowp
    int np = -1;
    A_rowp[0] = 0;
    B_rowp[0] = 0;

    for (int i = 0; i < A_rows; i++) {
      A_rowp[i + 1] = A_rowp[i + 1] + A_rowp[i];
      B_rowp[i + 1] = B_rowp[i + 1] + B_rowp[i];
      if (B_rowp[i + 1] > 0 && np == -1) {
        np = i;
      }
    }
    if (np == -1) {
      np = A_rows;
    }

    int nc = A_rows - np;
    if (np > 0) {
      int *temp = new int[nc + 1];
      for (int i = 0; i < nc + 1; i++) {
        temp[i] = B_rowp[i + np];
      }
      delete[] B_rowp;
      B_rowp = temp;
    }

    // Now, go through and build up A_cols/B_cols
    int *A_cols = new int[A_rowp[A_rows]];
    int *B_cols = new int[B_rowp[nc]];

    // Size the A_rowp/B_rowp arrays
    for (int i = 0; i < A_rows; i++) {  // For each variable
      int A_row_size = 0;
      int B_row_size = 0;

      int ei = var_map[i];  // Index into the external variable map
      int var = lower + i;

      if (ei >= 0) {
        // Add variables in this range to the row as well
        int start = rowp[ei];
        int end = rowp[ei + 1];

        for (int j = start; j < end; j++) {
          int col_var = ext_vars[cols[j]];
          if (col_var >= lower && col_var < upper) {
            A_row_vars[A_row_size] = col_var - lower;
            A_row_size++;
          } else {
            B_row_vars[B_row_size] = col_var;
            B_row_size++;
          }
        }
      }

      // Merge the off-processor contributions to the rows of A/B
      for (int k = 0; k < num_in_procs; k++) {
        // Try to find the variable in the k-th input - these are sorted
        // locally between in_rows[in_row_ptr[k]:in_row_ptr[k+1]]
        int count = in_row_ptr[k + 1] - in_row_ptr[k];
        int *item = TacsSearchArray(var, count, &in_rows[in_row_ptr[k]]);

        if (item) {
          int row = item - &in_rows[in_row_ptr[k]];
          row += in_row_ptr[k];

          // Add variables in this range to the row as well
          int start = in_rowp[row];
          int end = in_rowp[row + 1];

          for (int j = start; j < end; j++) {
            int col_var = in_cols[j];
            if (col_var >= lower && col_var < upper) {
              A_row_vars[A_row_size] = col_var - lower;
              A_row_size++;
            } else {
              B_row_vars[B_row_size] = col_var;
              B_row_size++;
            }
          }
        }
      }

      // Sort the entries and remove duplicates
      A_row_size = TacsUniqueSort(A_row_size, A_row_vars);
      for (int j = A_rowp[var - lower], k = 0; k < A_row_size; j++, k++) {
        A_cols[j] = A_row_vars[k];
      }

      // Convert the global indices into the local ordering
      B_row_size = TacsUniqueSort(B_row_size, B_row_vars);
      if (var - lower >= np) {
        for (int k = 0; k < B_row_size; k++) {
          int *item =
              TacsSearchArray(B_row_vars[k], col_map_size, col_map_vars);

          if (!item) {
            fprintf(stderr, "[%d] Error: variable %d not in column map\n",
                    mpiRank, B_row_vars[k]);
          } else {
            B_row_vars[k] = item - col_map_vars;
          }
        }

        int index = var - lower - np;
        for (int j = B_rowp[index], k = 0; k < B_row_size; j++, k++) {
          B_cols[j] = B_row_vars[k];
        }
      }
    }

    delete[] A_row_vars;
    delete[] B_row_vars;
    delete[] var_map;

    *_Arowp = A_rowp;
    *_Acols = A_cols;
    *_np = np;
    *_Browp = B_rowp;
    *_Bcols = B_cols;
  }

  /*!
    Given a non-zero pattern, pass in the values for the array
  */
  void TACSMatDistribute::setValues(TACSParallelMat *mat, int nvars,
                                    const int *ext_vars, const int *rowp,
                                    const int *cols, TacsScalar *avals) {
    // Get the block matrices
    BCSRMat *Aloc, *Bext;
    mat->getBCSRMat(&Aloc, &Bext);

    // Get the number of local variables and number of coupling
    // variables
    int N, Nc;
    mat->getRowMap(NULL, &N, &Nc);
    int Np = N - Nc;

    int bsize = Aloc->getBlockSize();
    int b2 = bsize * bsize;

    // Determine the maximum size of the array
    int max_row = 0;
    for (int i = 0; i < nvars; i++) {
      int size = rowp[i + 1] - rowp[i];
      if (size > max_row) {
        max_row = size;
      }
    }

    int mpiRank;
    MPI_Comm_rank(comm, &mpiRank);

    const int *ownerRange;
    row_map->getOwnerRange(&ownerRange);

    int lower = ownerRange[mpiRank];
    int upper = ownerRange[mpiRank + 1];

    int *acols = new int[2 * max_row];
    int *bcols = &acols[max_row];

    for (int i = 0; i < nvars; i++) {
      int row = ext_vars[i];

      // Check if this row is in the local or external block
      int nb = 0;
      if (row >= lower && row < upper) {
        row = row - lower;

        // Convert the cols
        int start = rowp[i];
        int end = rowp[i + 1];
        for (int j = rowp[i], k = 0; j < end; j++, k++) {
          int col = ext_vars[cols[j]];
          acols[k] = -1;
          bcols[k] = -1;

          if (col >= lower && col < upper) {
            acols[k] = col - lower;
          } else {
            int *item = TacsSearchArray(col, col_map_size, col_map_vars);
            bcols[k] = item - col_map_vars;
            nb++;
          }
        }

        Aloc->addBlockRowValues(row, end - start, acols, &avals[b2 * start]);

        if (nb > 0) {
          row = row - Np;
          if (row >= 0 && row < Nc) {
            Bext->addBlockRowValues(row, end - start, bcols,
                                    &avals[b2 * start]);
          } else {
            fprintf(stderr, "[%d] DistMat error: could not find row %d\n",
                    mpiRank, row);
          }
        }
      } else {
        int *item = TacsSearchArray(row, num_ext_rows, ext_rows);

        if (item) {
          int r_ext = item - ext_rows;

          int end = rowp[i + 1];
          for (int j = rowp[i], k = 0; j < end; j++, k++) {
            int c = cols[j];
            if (c >= 0 && c < nvars) {
              int col = ext_vars[c];

              int ext_start = ext_rowp[r_ext];
              int ext_size = ext_rowp[r_ext + 1] - ext_start;
              item = TacsSearchArray(col, ext_size, &ext_cols[ext_start]);

              if (item) {
                TacsScalar *a = &ext_A[b2 * (item - ext_cols)];
                memcpy(a, &avals[b2 * j], b2 * sizeof(TacsScalar));
              } else {
                fprintf(stderr,
                        "[%d] DistMat error: could not find col "
                        "(%d,%d) r_ext = %d \n",
                        mpiRank, row, col, r_ext);
              }
            } else {
              fprintf(stderr,
                      "[%d] DistMat error: local column out of "
                      "range 0 <= %d < %d\n",
                      mpiRank, c, nvars);
            }
          }
        } else {
          fprintf(stderr, "[%d] DistMat error: could not find row %d\n",
                  mpiRank, row);
        }
      }
    }

    delete[] acols;
  }

  /*
     Initiate the communication of the off-process matrix entries
  */
  void beginAssembly(TACSParallelMat *mat) {
    int mpiRank;
    MPI_Comm_rank(comm, &mpiRank);

    // Get the block size squared
    const int b2 = bsize * bsize;

    // Post the recvs
    for (int k = 0, offset = 0, buff_offset = 0; k < num_in_procs; k++) {
      int count = in_rowp[offset + in_count[k]] - in_rowp[offset];
      count *= b2;

      int source = in_procs[k];
      int tag = 5;
      MPI_Irecv(&in_A[buff_offset], count, TACS_MPI_TYPE, source, tag, comm,
                &in_requests[k]);
      offset += in_count[k];
      buff_offset += count;
    }

    // Post the sends
    for (int k = 0, offset = 0, buff_offset = 0; k < num_ext_procs; k++) {
      int count = ext_rowp[offset + ext_count[k]] - ext_rowp[offset];
      count *= b2;

      int dest = ext_procs[k];
      int tag = 5;
      MPI_Isend(&ext_A[buff_offset], count, TACS_MPI_TYPE, dest, tag, comm,
                &ext_requests[k]);
      offset += ext_count[k];
      buff_offset += count;
    }
  }

  /*
    Finish the communication of the off-process matrix entries.
    Once the communication is completed, add the off-processor
    entries to the matrix.
  */
  void endAssembly(TACSParallelMat *mat) {
    int mpiRank;
    MPI_Comm_rank(comm, &mpiRank);

    // Get the map between the global-external variables and the local
    // variables (for Bext)
    BCSRMat *Aloc, *Bext;
    mat->getBCSRMat(&Aloc, &Bext);

    // Get the number of local variables and number of coupling
    // variables
    int N, Nc;
    mat->getRowMap(NULL, &N, &Nc);
    int Np = N - Nc;

    // Get the block size squared
    const int b2 = bsize * bsize;

    // Find the owner range for mapping variables
    const int *ownerRange;
    row_map->getOwnerRange(&ownerRange);
    int lower = ownerRange[mpiRank];
    int upper = ownerRange[mpiRank + 1];

    for (int i = 0; i < num_in_procs; i++) {
      // Get the recv that just completed
      int index;
      MPI_Status status;
      int ierr = MPI_Waitany(num_in_procs, in_requests, &index, &status);

      // Check whether the recv was successful
      if (ierr != MPI_SUCCESS) {
        int len;
        char err_str[MPI_MAX_ERROR_STRING];
        MPI_Error_string(ierr, err_str, &len);
        fprintf(stderr,
                "[%d] TACSMatDistribute::endAssembly MPI_Waitany "
                "error\n%s\n",
                mpiRank, err_str);
      }

      // Identify which group of rows were just recv'd from another
      // processor
      for (int j = in_row_ptr[index]; j < in_row_ptr[index + 1]; j++) {
        // Find the local index of the row
        int row = in_rows[j] - lower;

        // Find the local row index
        for (int k = in_rowp[j]; k < in_rowp[j + 1]; k++) {
          TacsScalar *a = &in_A[b2 * k];

          // Set the column indices
          int col = in_cols[k];
          if (col >= lower && col < upper) {
            // Get the local column index
            col = col - lower;
            Aloc->addBlockRowValues(row, 1, &col, a);
          } else {
            // Use the map from the global column index back to the
            // processor
            int *item = TacsSearchArray(col, col_map_size, col_map_vars);
            if (item) {
              int c = item - col_map_vars;
              Bext->addBlockRowValues(row - Np, 1, &c, a);
            }
          }
        }
      }
    }

    // Wait for all the sending requests
    if (num_ext_procs > 0) {
      MPI_Waitall(num_ext_procs, ext_requests, MPI_STATUSES_IGNORE);
    }
  }
};

}  // namespace amigo

#endif  // AMIGO_CSR_DISTRIBUTE_H