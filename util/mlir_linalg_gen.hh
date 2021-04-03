#ifndef __MLIR_LINALG_GEN_HH__
#define __MLIR_LINALG_GEN_HH__

#include <fstream>

#include "../model.hh"
#include "mlir_linalg_dict.hh"

namespace SODA_FrontEnd
{
namespace Linalg
{
class MLIRGen
{
  public:
    typedef Model::Layer Layer;

  protected:
    std::ofstream mlir;

    // Need to track of register ID assignment
    uint64_t global_register_tracker = 0;
    // Track the output buffer ID of each layer
    std::vector<uint64_t> layer_output_buffer;

    // dictionary
    MLIRDict dict;

  public:
    MLIRGen(std::string &_fn) { mlir.open(_fn); }

    void genInit();

    void genInputLayer(Layer& layer);
    void genConv2DLayer(Layer& prev_layer,
                        Layer& cur_layer);
    void genEnd();

  protected:

    std::string genMemRef(std::vector<unsigned> &dims,
                          Layer::Data_Type &d_type);

    std::string genDilations(std::vector<unsigned>&);
    std::string genPaddings(std::vector<std::vector<unsigned>>&);
    std::string genStrides(std::vector<unsigned>&);
};
}
}

#endif
