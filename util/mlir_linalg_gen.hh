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

    void genInit()
    {
        mlir << "// AutoGen - Do not modify\n"
             << "func @main() -> ()\n"
             << "{\n"
             << "    // Global register id starts at: "
             << global_register_tracker << "\n";
    }

    void genInputLayer(Layer& layer)
    {
        auto& input_shape = layer.getOutputDim();

        mlir << "    // Layer Type: Input\n"
             << "    // Layer Name: " << layer.getName() << "\n"
             << "    // Input from Prev. Layer: nan\n"
             << "    // Input Size: ";
        for (auto dim : input_shape) { mlir << dim << " "; }
        mlir << "\n";

        std::string code = "    %" + std::to_string(global_register_tracker)
                         + " = "
                         + dict[ALLOC]
                         + "() : "
                         + genMemRef(layer);

	mlir << code << "\n";
        layer_output_buffer.push_back(global_register_tracker++);
        mlir << "\n";
    }

    void genConv2DLayer(Layer& prev_layer,
                        Layer& cur_layer)
    {
    
    }

    void genEnd()
    {
        mlir << "    return;\n"
             << "}\n";
        mlir.close();
    }

  protected:

    std::string genMemRef(Layer& layer)
    {
        auto& dim = layer.getOutputDim();
        std::string ret = "memref<";
        for (auto dim : dim)
        {
            ret += std::to_string(dim);
            ret += "x";
        }

        if (layer.isF32())
        {
            ret += "f32>";
        }

        return ret;
    }
};
}
}

#endif
