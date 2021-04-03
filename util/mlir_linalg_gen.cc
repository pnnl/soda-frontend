#include "mlir_linalg_gen.hh"

namespace SODA_FrontEnd
{
namespace Linalg
{
void MLIRGen::genInit()
{
    mlir << "// AutoGen - Do not modify\n"
         << "func @main() -> ()\n"
         << "{\n"
         << "    // Global register id starts at: "
         << global_register_tracker << "\n";
}

void MLIRGen::genInputLayer(Layer& layer)
{
    auto& input_shape = layer.getOutputDim();
    auto& input_dtype = layer.getDataType();

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
                     + genMemRef(input_shape, input_dtype);

    mlir << code << "\n";
    layer_output_buffer.push_back(global_register_tracker++);
    mlir << "\n";
}

void MLIRGen::genConv2DLayer(Layer& prev_layer,
                             Layer& cur_layer)
{
    mlir << "    // Layer Type: Conv2D\n"
         << "    // Layer Name: " << cur_layer.getName() << "\n";

    mlir << "\n";
}

void MLIRGen::genEnd()
{
    mlir << "    return;\n"
         << "}\n";
    mlir.close();
}

std::string MLIRGen::genMemRef(std::vector<unsigned> &dims,
                               Layer::Data_Type &d_type)
{
    std::string ret = "memref<";
    for (auto dim : dims)
    {
        ret += std::to_string(dim);
        ret += "x";
    }

    if (d_type == Layer::Data_Type::f32)
    {
        ret += "f32>";
    }

    return ret;
}
}
}

