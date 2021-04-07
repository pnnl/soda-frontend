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

  protected:
    std::vector<std::string> index_str;

  public:
    MLIRGen(std::string &_fn) 
    { 
        index_str = 
            {"a", "b", "c", "d", "e", "f", "g", "h", "i",
             "j", "k", "l", "m", "n", "o", "p", "q", "r",
             "s", "t", "u", "v", "w", "x", "y", "z"};

        mlir.open(_fn); 
    }

    void genInit();

    void genInputLayer(Layer& layer);
    void genConv2DLayer(Layer& prev_layer,
                        Layer& cur_layer);
    void genActLayer(Layer& prev_layer,
                     Layer& cur_layer);
    void genMaxPooling2DLayer(Layer& prev_layer,
                              Layer& cur_layer);

    void genFlattenLayer(Layer& prev_layer,
                         Layer& cur_layer);

    void genEnd();

  protected:

    std::string genMemRef(std::vector<unsigned> &dims,
                          Layer::Data_Type &d_type);

    std::string genDilations(std::vector<unsigned>&);

    std::string genPaddings(std::vector<std::vector<unsigned>>&);

    std::string genStrides(std::vector<unsigned>&);

    std::string genRelu(unsigned,
                        std::vector<unsigned>&,
                        std::string&,
                        std::string&);

    std::string genLoad(unsigned,unsigned,unsigned,std::string&);

    std::string genStore(std::string&,unsigned,unsigned,unsigned,std::string&);

    std::string genZeroF()
    {
        return "%zero = constant 0.00000e+00 : f32";
    }
};
}
}

#endif
