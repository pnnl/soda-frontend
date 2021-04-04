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
// TODO-Shihao, fill kernel into memory
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
    cur_layer.setOutputDim(output_shape);

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

void MLIRGen::genActLayer(Layer& prev_layer,
                          Layer& cur_layer)
{
    mlir << "    // Layer Type: Activation\n"
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

    // Output shape reminds the same
    mlir << "    // Output buffer: %"
         << input_buffer_reg << " : "
         << input_memref << "\n";

    cur_layer.setOutputDim(input_shape);
    // Keep the same 
    layer_output_buffer.push_back(input_buffer_reg);

    std::string code;
    if (cur_layer.getActivation() == Layer::Activation::relu)
    {
        mlir << "    // Activation: relu\n";
        std::string cur_layer_dtype;
        if (cur_layer.getDataType() == Layer::Data_Type::f32)
        {
            cur_layer_dtype = "f32";
        }
        code = genRelu(input_buffer_reg, 
                       input_shape, 
                       input_memref, 
                       cur_layer_dtype);
    }
    mlir << code;
    mlir << "\n";
}

// TODO-Shihao, should support padding in the future
void MLIRGen::genMaxPooling2DLayer(Layer& prev_layer,
		                   Layer& cur_layer)
{
    mlir << "    // Layer Type: MaxPooling2D\n"
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

    auto& output_dtype = cur_layer.getDataType();
    std::vector<unsigned> output_shape;

    unsigned out_height, out_width;
    if (cur_layer.padding_type == Layer::Padding_Type::valid)
    {
        out_height = ceil(float(input_shape[1] - kernel[0] + 1) / 
                     float(stride[0]));
	
        out_width = ceil(float(input_shape[2] - kernel[1] + 1) / 
                     float(stride[1]));
    }
    
    output_shape.push_back(input_shape[0]);
    output_shape.push_back(out_height);
    output_shape.push_back(out_width);
    output_shape.push_back(input_shape[3]);
    mlir << "    // Output size: ";
    for (auto dim : output_shape) { mlir << dim << " "; }
    mlir << "\n";
    cur_layer.setOutputDim(output_shape);

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

    // Generate linalg maxpooling dialect
    stride.insert(stride.begin(), 1);
    stride.push_back(1);
    code = "    " + dict[MAXPOOL] + "("
         + "%" + std::to_string(input_buffer_reg) + ", "
         + "%" + std::to_string(kernel_reg) + ", "
         + "%" + std::to_string(output_reg) + ")\n"
         + "    {\n"
         + "        " + genStrides(stride) + "\n"
         + "    } : "
         + input_memref + ", "
         + kernel_memref + ", "
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

    // TODO, there should be lots error checking through the codes
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

std::string MLIRGen::genRelu(unsigned buffer_id,
                             std::vector<unsigned> &shape,
                             std::string &shape_memref,
                             std::string &dtype)
{
    std::string ret;

    int num_of_loops = shape.size();
    
    // Gen loop statement
    for (auto i = 0; i < num_of_loops; i++)
    {
        std::string one_loop = 
            std::string(4 + i * 2, ' ')
            + dict[FOR] 
            + " %" + index_str[i] + " = 0 to " 
            + std::to_string(shape[i])
            + " step 1\n"
            + std::string(4 + i * 2, ' ') + "{\n";
        ret += one_loop; 
    }

    // Gen loading
    auto load_str = std::string(4 + num_of_loops * 2, ' ')
                  + "%tmp = "
                  + genLoad(buffer_id, 0, num_of_loops - 1, shape_memref) + "\n";
    ret += load_str;

    // Gen zero
    std::string zero = "\%zero";
    if (dtype == "f32")
    {
        ret += (std::string(4 + num_of_loops * 2, ' ') + 
                genZeroF() + "\n");
    }

    // Gen compare
    ret += (std::string(4 + num_of_loops * 2, ' ') + 
           "\%cond = " + dict[CMPLT] + ", %tmp, \%zero : " + 
           dtype + "\n");

    // Gen if-store
    ret += (std::string(4 + num_of_loops * 2, ' ') + 
            dict[IF] + " \%cond\n" +
            std::string(4 + num_of_loops * 2, ' ') + "{\n" +
            std::string(4 + (num_of_loops + 1) * 2, ' ') +
            genStore(zero, buffer_id, 0, num_of_loops - 1, shape_memref) + "\n" +
            std::string(4 + num_of_loops * 2, ' ') + "}\n");

    // Gen loop end
    for (auto i = num_of_loops - 1; i >= 0; i--)
    {
        std::string one_loop = 
            std::string(4 + i * 2, ' ') + "}\n";
        ret += one_loop; 
    }
    return ret;
}

std::string MLIRGen::genLoad(unsigned buffer_id,
                             unsigned index_start,
                             unsigned index_end,
                             std::string& mem_ref)
{
    std::string ret = dict[LOAD] + " %" + std::to_string(buffer_id) + "[";
    for (auto i = index_start; i < index_end; i++)
    {
        ret += ("%" + index_str[i] + ", ");
    }
    ret += ("%" + index_str[index_end] + "] : " + mem_ref);

    return ret;
}

std::string MLIRGen::genStore(std::string &val,
                              unsigned buffer_id,
                              unsigned index_start,
                              unsigned index_end,
                              std::string& mem_ref)
{
    std::string ret = dict[STORE] + " " + val + ", "
                    + "%" + std::to_string(buffer_id) + "[";
    for (auto i = index_start; i < index_end; i++)
    {
        ret += ("%" + index_str[i] + ", ");
    }
    ret += ("%" + index_str[index_end] + "] : " + mem_ref);

    return ret;
}


}
}

