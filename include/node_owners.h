#ifndef AMIGO_NODE_OWNERS_H
#define AMIGO_NODE_OWNERS_H

#include <mpi.h>

namespace amigo {

/**
 * @brief Store the node to processor assignments and the implicit mapping from
 * the local indices to the global indices
 *
 * The global node numbers on each processor are mapped to local indices.
 *
 *
 */
class NodeOwners {
 public:
  NodeOwners(MPI_Comm comm, const int range_[], int num_ext_nodes = 0,
             const int ext_nodes_[] = nullptr)
      : comm(comm), num_ext_nodes(num_ext_nodes) {
    int size;
    MPI_Comm_size(comm, &size);
    range = new int[size + 1];
    for (int i = 0; i < size + 1; i++) {
      range[i] = range_[i];
    }
    ext_nodes = new int[num_ext_nodes];
    for (int i = 0; i < num_ext_nodes; i++) {
      ext_nodes[i] = ext_nodes_[i];
    }
  }
  ~NodeOwners() {
    delete[] range;
    if (ext_nodes) {
      delete[] ext_nodes;
    }
  }

  /**
   * @brief Get the MPI communicator object
   */
  MPI_Comm get_mpi_comm() const { return comm; }

  /**
   * Get the range of nodes owned by each processor
   */
  const int *get_range() const { return range; }

  /**
   * @brief Get the number of local variables
   */
  int get_local_size() const {
    int rank;
    MPI_Comm_rank(comm, &rank);
    return range[rank + 1] - range[rank];
  }

  /**
   * @brief Get the number of local
   */
  int get_ext_size() const { return num_ext_nodes; }

  /**
   * @brief Get the external nodes
   */
  int get_ext_nodes(const int *nodes[]) const {
    if (nodes) {
      *nodes = ext_nodes;
    }
    return num_ext_nodes;
  }

 private:
  MPI_Comm comm;
  int *range;

  // Store mapping of the local node numbers
  int num_ext_nodes;
  int *ext_nodes;
};

}  // namespace amigo

#endif  // AMIGO_NODE_OWNERS_H