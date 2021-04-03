#include "mlir_linalg_gen.hh"

#include <cmath>
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

    mlir << "    // Layer type: Input\n"
         << "    // Layer tame: " << layer.getName() << "\n"
         << "    // Input from Prev. layer: nan\n"
         << "    // Input size: ";
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

// TODO, this is only for 2D image
// TODO-Shihao, discuss with Vinay on what happens for a 3D image
void MLIRGen::genConv2DLayer(Layer& prev_layer,
                             Layer& cur_layer)
{
    mlir << "    // Layer Type: Conv2D\n"
         << "    // Layer Name: " << cur_layer.getName() << "\n";
    auto prev_layer_id = prev_layer.getID();
    auto cur_layer_id = cur_layer.getID();
    mlir << "    // Input from layer: " << prev_layer.getName() << "\n";
    auto input_buffer_reg = layer_output_buffer[prev_layer_id];
    mlir << "    // Input buffer: %"
         << input_buffer_reg << " : ";
    auto& input_shape = prev_layer.getOutputDim();
    auto& input_dtype = prev_layer.getDataType();
    auto input_memref = genMemRef(input_shape, input_dtype);
    mlir << input_memref << "\n";

    auto &kernel = cur_layer.getKernelDim();
    mlir << "    // Kernel dim.: ";
    for (auto dim : kernel) { mlir << dim << " "; }
    mlir << "\n";

    auto &stride = cur_layer.getStrides();
    mlir << "    // Stride dim.: ";
    for (auto dim : stride) { mlir << dim << " "; }
    mlir << "\n";

    auto &dilation = cur_layer.getDilations();
    mlir << "    // Dilation rates: ";
    for (auto rate : dilation) { mlir << rate << " "; }
    mlir << "\n";

    // Determine output dimension and paddings
    // filter(nH, nW, nC, nF)
    // in (nN, nH, nW, nC)
    // out (nN, nH, nW, nC) 
    auto& output_dtype = cur_layer.getDataType();
    std::vector<unsigned> output_shape;
    std::vector<std::vector<unsigned>> padding;

    unsigned out_height, out_width;
    unsigned pad_top = 0, pad_bottom = 0, 
             pad_left = 0, pad_right = 0;
    if (cur_layer.padding_type == Layer::Padding_Type::valid)
    {
        out_height = ceil(float(input_shape[1] - kernel[0] + 1) / 
                     float(stride[0]));
	
        out_width = ceil(float(input_shape[2] - kernel[1] + 1) / 
                     float(stride[1]));
    }
    else if (cur_layer.padding_type == Layer::Padding_Type::same)
    {
        out_height = ceil(float(input_shape[1]) / float(stride[0]));
	
        out_width = ceil(float(input_shape[2]) / float(stride[1]));

        unsigned pad_along_height = std::max(int((out_height - 1) * stride[0] 
                                  + kernel[0] - input_shape[1]), 0);

        unsigned pad_along_width = std::max(int((out_width - 1) * stride[1] 
                                 + kernel[1] - input_shape[2]), 0);
        // std::cout << pad_along_height << " " << pad_along_width << "\n";
        pad_top	= pad_along_height / 2;
        pad_bottom = pad_along_height - pad_top;
        pad_left = pad_along_width / 2;
        pad_right = pad_along_width - pad_left;
    }
    output_shape.push_back(input_shape[0]);
    output_shape.push_back(out_height);
    output_shape.push_back(out_width);
    output_shape.push_back(kernel[3]);
    mlir << "    // Output size: ";
    for (auto dim : output_shape) { mlir << dim << " "; }
    mlir << "\n";

    std::vector<unsigned> pad_along_height = 
        {pad_top, pad_bottom};
    std::vector<unsigned> pad_along_width = 
        {pad_left, pad_right};
    padding = {pad_along_height, pad_along_width};
    mlir << "    // Padding: ";
    for (auto dir : padding)
    {
        mlir << "[ ";
        for (auto p : dir) { mlir << p << " "; }
        mlir << "] ";
    }
    mlir << "\n";
    // real code generation
    // alloc kernel
    auto &cur_layer_dtype = cur_layer.getDataType();
    auto kernel_reg = global_register_tracker;
    auto kernel_memref = genMemRef(kernel, cur_layer_dtype);
    std::string code = "    %" + std::to_string(kernel_reg)
                     + " = "
                     + dict[ALLOC]
                     + "() : "
                     + kernel_memref;
    mlir << code << "\n";
    global_register_tracker++;

    // alloc output
    auto output_reg = global_register_tracker;
    auto output_memref = genMemRef(output_shape, cur_layer_dtype);
    code = "    %" + std::to_string(output_reg)
                   + " = "
                   + dict[ALLOC]
                   + "() : "
                   + output_memref;
    mlir << code << "\n";
    layer_output_buffer.push_back(global_register_tracker);
    global_register_tracker++;

    // generate linalg operation
    code = "    " + dict[CONV2D] + "("
         + "%" + std::to_string(kernel_reg) + ", "
         + "%" + std::to_string(input_buffer_reg) + ", "
         + "%" + std::to_string(output_reg) + ")\n"
         + "    {\n"
         + "        " + genDilations(dilation) + ",\n"
         + "        " + genPaddings(padding) + ",\n"
         + "        " + genStrides(stride) + "\n"
         + "    } : "
         + kernel_memref + ", "
         + input_memref + ", "
         + output_memref + "\n";
    mlir << code;
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

std::string MLIRGen::genDilations(std::vector<unsigned> &dilation)
{
    std::string ret = "";
    if (dilation.size() == 0) return ret;

    ret = "dilations = [";
    for (auto i = 0; i < dilation.size() - 1; i++)
    {
        ret += std::to_string(dilation[i]) + ", ";
    }
    ret += std::to_string(dilation[dilation.size() - 1]) + "]";
    return ret;
}

std::string MLIRGen::genPaddings(std::vector<std::vector<unsigned>> &padding)
{
    std::string ret = "";
    if (padding.size() == 0) return ret;

    ret = "padding = dense<";
    auto out_cnt = 0;
    for (auto dir : padding)
    {
        ret += "[";
        auto in_cnt = 0;
        for (auto p : dir)
        {
            if (in_cnt < (dir.size() - 1))
            {
                ret += std::to_string(p) + ", ";
            }
            else
            {
                ret += std::to_string(p);
            }
            in_cnt++;
        }
        if (out_cnt < (padding.size() - 1))
        {
            ret += "], ";
        }
        else
        {
            ret += "]";
        }
        out_cnt++;
    }

    ret = ret + "> : tensor<"
        + std::to_string(padding.size()) + "x"
        + std::to_string(padding[0].size()) + "x"
        + "i64>"; // TODO, hard-coded
    return ret;
}

std::string MLIRGen::genStrides(std::vector<unsigned>& stride)
{
    std::string ret = "";
    if (stride.size() == 0) return ret;

    ret = "strides = [";
    for (auto i = 0; i < stride.size() - 1; i++)
    {
        ret += std::to_string(stride[i]) + ", ";
    }
    ret += std::to_string(stride[stride.size() - 1]) + "]";
    return ret;

}
}
}

