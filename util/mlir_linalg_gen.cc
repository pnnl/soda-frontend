// Copyright 2021 Battelle Memorial Institute

#include "mlir_linalg_gen.hh"

#include <cassert>
#include <cmath>
#include <cassert>
namespace SODA_FrontEnd
{
namespace Linalg
{

void MLIRGen::genLayerBody(std::vector<Layer>& layers, unsigned layer_id) 
{
    if (layers[layer_id].layer_type == Layer::Layer_Type::Input)
    {
            genInputLayer(layers[layer_id]);
    }
    else if (layers[layer_id].layer_type == Layer::Layer_Type::Conv2D)
    {
        genConv2DLayer(layers[layer_id - 1], layers[layer_id]); 
    }
    else if (layers[layer_id].layer_type == Layer::Layer_Type::Activation)
    {
        if(layers[layer_id].activation == Layer::Activation::relu)
            genActLayer(layers[layer_id - 1], layers[layer_id]);
        else if(layers[layer_id].activation == Layer::Activation::softmax)
            genSoftMaxLayer(layers[layer_id - 1], layers[layer_id]);
    }
    else if (layers[layer_id].layer_type == Layer::Layer_Type::MaxPooling2D)
    {
        genMaxPooling2DLayer(layers[layer_id - 1], layers[layer_id]);
    }
    else if (layers[layer_id].layer_type == Layer::Layer_Type::Flatten)
    {
        genFlattenLayer(layers[layer_id - 1], layers[layer_id]);
    }
    else if (layers[layer_id].layer_type == Layer::Layer_Type::Dense)
    {
        genDenseLayer(layers[layer_id - 1], layers[layer_id]);
    }
    else if (layers[layer_id].layer_type == Layer::Layer_Type::BatchNormalization)
    {
        // For Convolution
        genBatchNormalizationLayer(layers[layer_id - 1], layers[layer_id]);
    }
    else if (layers[layer_id].layer_type == Layer::Layer_Type::ZeroPadding2D)
    {
        // For Convolution
        genZeroPadding2DLayer(layers[layer_id - 1], layers[layer_id]);
    }
    else if (layers[layer_id].layer_type == Layer::Layer_Type::Add)
    {
        // For Convolution Add
        genAddLayer(layers[layer_id - 1], layers[layer_id]);
    }
    else if (layers[layer_id].layer_type == Layer::Layer_Type::GlobalAveragePooling2D)
    {
        // For Convolution Add, overlap with MaxPooling/AveragePooling?? 
        genGlobalAveragePooling2DLayer(layers[layer_id - 1], layers[layer_id]);
    }

}
void MLIRGen::genKernelLoad(Layer& layer) 
{
    if (layer.getLayerType() == Layer::Layer_Type::Conv2D)
    {
        auto kernel_dim = layer.getKernelDim();
        auto cur_layer_dtype = layer.getDataType();
        auto kernel_memref = genMemRef(kernel_dim,
                                       cur_layer_dtype);
        auto tensor_const = genTensorConstF4D(layer.getKernel(),
                                              kernel_dim,
                                              cur_layer_dtype);

        auto var_name = "@" + layer.getName() + "_kernel";

        mlir << "memref.global \"private\" constant "
             << var_name << " : " << kernel_memref
             << " = dense<" << tensor_const << ">\n\n";
    }
    else if (layer.getLayerType() == Layer::Layer_Type::BatchNormalization) 
    {
        auto cur_layer_dtype = layer.getDataType();
        auto gamma_dim = layer.getGammaDim();
        auto gamma_memref = genMemRef(gamma_dim,
                                       cur_layer_dtype);
        auto tensor_const_gamma = genTensorConstF1D(layer.getGamma(),
                                              gamma_dim,
                                              cur_layer_dtype);
        
        auto var_name = "@" + layer.getName() + "_gamma";

        mlir << "memref.global \"private\" constant "
             << var_name << " : " << gamma_memref
             << " = dense<" << tensor_const_gamma << ">\n\n";

        auto beta_dim = layer.getBetaDim();
        auto beta_memref = genMemRef(beta_dim, cur_layer_dtype);
        auto tensor_const_beta = genTensorConstF1D(layer.getBeta(),
                                              beta_dim,
                                              cur_layer_dtype);
        auto var_name_beta = "@" + layer.getName() + "_beta";

        mlir << "memref.global \"private\" constant "
             << var_name_beta << " : " << beta_memref
             << " = dense<" << tensor_const_beta << ">\n\n";                                                                     
    }

}

// TODO-Shihao
// We temporarily put all the weights/bias as global private constants
void MLIRGen::genInitLayerTest(Layer& layer)
{
    mlir << "// AutoGen - Do not modify\n\n";
    genKernelLoad(layer);
    mlir << "func @main() -> ()\n"
         << "{\n"
         << "    // Global register id starts at: "
         << global_register_tracker << "\n";
}

void MLIRGen::genInit(std::vector<Layer> &layers)
{
    mlir << "// AutoGen - Do not modify\n\n";

    for (auto i = 0; i < layers.size(); i++)
    {
        genKernelLoad(layers[i]);
    }

    mlir << "func @main() -> ()\n"
         << "{\n"
         << "    // Global register id starts at: "
         << global_register_tracker << "\n";
}

void MLIRGen::genInputLayer(Layer& layer)
{
    auto& input_shape = layer.getOutputDim();
    auto& input_dtype = layer.getDataType();

    mlir << "    // Layer type: Input\n"
         << "    // Layer name: " << layer.getName() << "\n"
         << "    // Input from Prev. layer: nan\n"
         << "    // Input size: ";
    for (auto dim : input_shape) { mlir << dim << " "; }
    mlir << "\n";

    const auto [it_var, flag] = variable_map.insert({layer.getName(),global_register_tracker++});
    if(!flag) {
        std::cout << "Variable map insertion failure" << std::endl;
    }
    // std::string code = "    %" + std::to_string(global_register_tracker)
    std::string code = "    %" + std::to_string(variable_map.at(layer.getName()))
                     + " = "
                     + dict[ALLOC]
                     + "() : "
                     + genMemRef(input_shape, input_dtype);

    mlir << code << "\n";
    // layer_output_buffer.push_back(global_register_tracker++);
    mlir << "\n";
}

// TODO, this is only for 2D image
// TODO-Shihao, discuss with Vinay on what happens for a 3D image
// TODO-Shihao, fill kernel into memory
void MLIRGen::genConv2DLayer(Layer& prev_layer,
                             Layer& cur_layer)
{
    mlir << "    // Layer type: Conv2D\n"
         << "    // Layer name: " << cur_layer.getName() << "\n";
    auto prev_layer_id = prev_layer.getID();
    auto cur_layer_id = cur_layer.getID();
    mlir << "    // Input from layer: " << prev_layer.getName() << "\n";
    auto input_buffer_reg = variable_map.at(prev_layer.getName());
    mlir << "    // Input buffer: %"
         << input_buffer_reg << " : ";
    auto& input_shape = prev_layer.getOutputDim();
    auto& input_dtype = prev_layer.getDataType();
    auto input_memref = genMemRef(input_shape, input_dtype);
    mlir << input_memref << "\n";
    
    if(this->isTest()) 
    {
        std::string code = "    %" + std::to_string(input_buffer_reg) 
                         + " = "
                         + dict[ALLOC]
                         + "() : "
                         + input_memref
                         + " \n";
        code += "    %c1 = constant 1.0 : f32 \n";                 
        code += "    " 
                + dict[FILLVAR]
                + "(%c1,%"
                + std::to_string(input_buffer_reg)
                + ") : f32, "
                + input_memref 
                + "\n";
        mlir << code << "\n";

    }

    auto &kernel_dim = cur_layer.getKernelDim();
    mlir << "    // Kernel dim.: ";
    for (auto dim : kernel_dim) { mlir << dim << " "; }
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
        // No padding
        out_height = floor(float(input_shape[1] - kernel_dim[0]) /
                     float(stride[0])) + 1;

        out_width = floor(float(input_shape[2] - kernel_dim[1]) /
                     float(stride[1])) + 1;
    }
    else if (cur_layer.padding_type == Layer::Padding_Type::same)
    {
        // Padding Present
        // TODO: consider other cases
        // Kernel size normally odd.
        unsigned padding_size = int(kernel_dim[0])/2;

        out_height =
            floor(float(input_shape[1] + 2 * padding_size - kernel_dim[0]) /
            float(stride[0])) + 1;

        out_width =
            floor(float(input_shape[2] + 2 * padding_size - kernel_dim[1]) /
            float(stride[1])) + 1;

        unsigned pad_along_height,pad_along_width;
        pad_along_height = pad_along_width = padding_size;
        pad_top	= pad_along_height / 2;
        pad_bottom = pad_along_height - pad_top;
        pad_left = pad_along_width / 2;
        pad_right = pad_along_width - pad_left;
    }
    output_shape.push_back(input_shape[0]);
    output_shape.push_back(out_height);
    output_shape.push_back(out_width);
    output_shape.push_back(kernel_dim[3]);
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
    auto kernel_name = cur_layer.getName() + "_kernel";
    const auto [it_var1, flag1] = variable_map.insert({kernel_name,global_register_tracker++});
    if(!flag1) {
        std::cout << "Variable map insertion failure" << std::endl;
    }
    auto kernel_reg = variable_map.at(kernel_name);
    auto kernel_memref = genMemRef(kernel_dim, cur_layer_dtype);
    std::string code = "    %" + std::to_string(kernel_reg)
                     + " = memref.get_global @"
                     + kernel_name
                     + " : "
                     + kernel_memref;
    mlir << code << "\n";

    // alloc output
    const auto [it_var2, flag2] = variable_map.insert({cur_layer.getName(),global_register_tracker++});
    if(!flag2) {
        std::cout << "Variable map insertion failure" << std::endl;
    }
    auto output_reg = variable_map.at(cur_layer.getName());
    auto output_memref = genMemRef(output_shape, cur_layer_dtype);
    code = "    %" + std::to_string(output_reg)
           + " = "
           + dict[ALLOC]
           + "() : "
           + output_memref;
    mlir << code << "\n";

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
    mlir << "    // Layer type: Activation\n"
         << "    // Layer name: " << cur_layer.getName() << "\n";
    auto prev_layer_id = prev_layer.getID();
    auto cur_layer_id = cur_layer.getID();
    mlir << "    // Input from layer: " << prev_layer.getName() << "\n";
    auto input_buffer_reg = variable_map.at(prev_layer.getName());
    mlir << "    // Input buffer: %"
         << input_buffer_reg << " : ";
    auto& input_shape = prev_layer.getOutputDim();
    auto& input_dtype = prev_layer.getDataType();
    auto input_memref = genMemRef(input_shape, input_dtype);
    mlir << input_memref << "\n";

    if(this->isTest()) 
    {
        std::string code = "    %" + std::to_string(input_buffer_reg) 
                         + " = "
                         + dict[ALLOC]
                         + "() : "
                         + input_memref
                         + " \n";
        code += "    %v1 = constant 1.0 : f32 \n";                 
        code += "    " 
                + dict[FILLVAR]
                + "(%v1,%"
                + std::to_string(input_buffer_reg)
                + ") : f32, "
                + input_memref 
                + "\n";
        mlir << code << "\n";

    }

    // Output shape reminds the same
    mlir << "    // Output buffer: %"
         << input_buffer_reg << " : "
         << input_memref << "\n";

    cur_layer.setOutputDim(input_shape);
    // Keep the same
    const auto [it_var, flag] = variable_map.insert({cur_layer.getName(),input_buffer_reg});
    if(!flag) {
        std::cout << "Variable map insertion failure" << std::endl;
    }

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
    mlir << "    // Layer type: MaxPooling2D\n"
         << "    // Layer name: " << cur_layer.getName() << "\n";
    auto prev_layer_id = prev_layer.getID();
    auto cur_layer_id = cur_layer.getID();
    mlir << "    // Input from layer: " << prev_layer.getName() << "\n";
    auto input_buffer_reg = variable_map.at(prev_layer.getName());
    mlir << "    // Input buffer: %"
         << input_buffer_reg << " : ";
    auto& input_shape = prev_layer.getOutputDim();
    auto& input_dtype = prev_layer.getDataType();
    auto input_memref = genMemRef(input_shape, input_dtype);
    mlir << input_memref << "\n";

    if(this->isTest()) 
    {
        std::string code = "    %" + std::to_string(input_buffer_reg) 
                         + " = "
                         + dict[ALLOC]
                         + "() : "
                         + input_memref
                         + " \n";
        code += "    %c1 = constant 1.0 : f32 \n";                 
        code += "    " 
                + dict[FILLVAR]
                + "(%c1,%"
                + std::to_string(input_buffer_reg)
                + ") : f32, "
                + input_memref 
                + "\n";
        mlir << code << "\n";

    }

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
    auto kernel_name = cur_layer.getName() + "_kernel";
    const auto [it_var1, flag1] = variable_map.insert({kernel_name,global_register_tracker++});
    if(!flag1) {
        std::cout << "Variable map insertion failure" << std::endl;
    }
    auto kernel_reg = variable_map.at(kernel_name);
    auto kernel_memref = genMemRef(kernel, cur_layer_dtype);
    std::string code = "    %" + std::to_string(kernel_reg)
                     + " = "
                     + dict[ALLOC]
                     + "() : "
                     + kernel_memref;
    mlir << code << "\n";

    // alloc output
    const auto [it_var2, flag2] = variable_map.insert({cur_layer.getName(),global_register_tracker++});
    if(!flag2) {
        std::cout << "Variable map insertion failure" << std::endl;
    }
    auto output_reg = variable_map.at(cur_layer.getName());
    auto output_memref = genMemRef(output_shape, cur_layer_dtype);
    code = "    %" + std::to_string(output_reg)
                   + " = "
                   + dict[ALLOC]
                   + "() : "
                   + output_memref;
    mlir << code << "\n";

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

void MLIRGen::genFlattenLayer(Layer& prev_layer,
                              Layer& cur_layer)
{
    mlir << "    // Layer type: Flatten\n"
         << "    // Layer name: " << cur_layer.getName() << "\n";
    auto prev_layer_id = prev_layer.getID();
    auto cur_layer_id = cur_layer.getID();
    mlir << "    // Input from layer: " << prev_layer.getName() << "\n";
    auto input_buffer_reg = variable_map.at(prev_layer.getName());
    mlir << "    // Input buffer: %"
         << input_buffer_reg << " : ";
    auto& input_shape = prev_layer.getOutputDim();

    std::cout << "Flatten Layer Input Dim: " << std::endl;
    for(auto i: input_shape)
        std::cout << i << " ";
    std::cout << std::endl;
    auto& input_dtype = prev_layer.getDataType();
    auto input_memref = genMemRef(input_shape, input_dtype);
    mlir << input_memref << "\n";

    if(this->isTest()) 
    {
        std::string code = "    %" + std::to_string(input_buffer_reg) 
                         + " = "
                         + dict[ALLOC]
                         + "() : "
                         + input_memref
                         + " \n";
        code += "    %c1 = constant 1.0 : f32 \n";                 
        code += "    " 
                + dict[FILLVAR]
                + "(%c1,%"
                + std::to_string(input_buffer_reg)
                + ") : f32, "
                + input_memref 
                + "\n";
        mlir << code << "\n";

    }

    // Determine output size
    auto &cur_layer_dtype = cur_layer.getDataType();
    unsigned out_size = 1;
    std::vector<unsigned> output_shape;
    for (int i = 1; i < input_shape.size(); ++i)
    {
        out_size *= input_shape[i];
    }
    output_shape.push_back(input_shape[0]);
    output_shape.push_back(out_size);
    mlir << "    // Output size: ";
    for (auto d: output_shape)
        mlir << d << " ";
    mlir << "\n";
    std::vector<unsigned> out_dim = {out_size};
    cur_layer.setOutputDim(output_shape);
    // cur_layer.setOutputDim(out_dim);

    const auto [it_var2, flag2] = variable_map.insert({cur_layer.getName(),global_register_tracker++});
    if(!flag2) {
        std::cout << "Variable map insertion failure" << std::endl;
    }
    auto out_buffer_reg = variable_map.at(cur_layer.getName());
    std::string out_memref = genMemRef(output_shape, cur_layer_dtype);
    std::string code = "    %" + std::to_string(out_buffer_reg)
                    + " = "
                    + dict[ALLOC]
                    + "() : "
                    + out_memref;
    mlir << code << "\n";

    // Generate flatten function
    code = "";
    int num_of_loops = input_shape.size();

    // Gen loop statement
    for (auto i = 0; i < num_of_loops; i++)
    {
        std::string one_loop =
            std::string(4 + i * 2, ' ')
            + dict[FOR]
            + " %" + default_index_str[i] + " = 0 to "
            + std::to_string(input_shape[i])
            + " step 1\n"
            + std::string(4 + i * 2, ' ') + "{\n";
        code += one_loop;
    }


    // Gen loading
    auto load_str = std::string(4 + num_of_loops * 2, ' ')
                  + "\%ld_val = "
                  + genLoad(default_index_str,
                            input_buffer_reg,
                            0,
                            num_of_loops,
                            // num_of_loops - 1,
                            input_memref) + "\n\n";
    code += load_str;

    // Gen output index
    code += (std::string(4 + num_of_loops * 2, ' ')
         + "\%index = "
         + dict[ADDI] + " \%zero, \%zero : i32\n\n");

    for (auto i = 0; i < num_of_loops - 1; i++)
    {
        auto size = 1;
        for (auto j = i + 1; j < num_of_loops; j++)
        {
            size *= input_shape[j];
        }

        code += (std::string(4 + num_of_loops * 2, ' ')
                + "\%index_tmp  = "
                + dict[MULI] + " "
                + "%" + default_index_str[i] + ", "
                + std::to_string(size) + " : i32\n");

        code +=  (std::string(4 + num_of_loops * 2, ' ')
                + "\%index = "
                + dict[ADDI] + " "
                + "\%index_tmp, "
                + "\%index" + " : i32\n\n");
    }
    code +=  (std::string(4 + num_of_loops * 2, ' ')
             + "\%index = "
             + dict[ADDI] + " "
             + "%" + default_index_str[num_of_loops - 1] + ", "
             + "\%index" + " : i32\n\n");



    // Gen store
    code += (std::string(4 + num_of_loops * 2, ' ') +
            dict[STORE] + " \%ld_val, %" +
            std::to_string(out_buffer_reg) +
            "[\%" + default_index_str[0]+ ",\%index] : " +
            out_memref + "\n");

    // Gen loop end
    for (auto i = num_of_loops - 1; i >= 0; i--)
    {
        std::string one_loop =
            std::string(4 + i * 2, ' ') + "}\n";
        code += one_loop;
    }

    mlir << code;

    mlir << "\n";
}

void MLIRGen::genDenseLayer(Layer& prev_layer,
                            Layer& cur_layer)
{
    mlir << "    // Layer type: Dense\n"
         << "    // Layer name: " << cur_layer.getName() << "\n";
    auto prev_layer_id = prev_layer.getID();
    auto cur_layer_id = cur_layer.getID();
    mlir << "    // Input from layer: " << prev_layer.getName() << "\n";
    auto input_buffer_reg = variable_map.at(prev_layer.getName());
    mlir << "    // Input buffer: %"
         << input_buffer_reg << " : ";
    auto& input_shape = prev_layer.getOutputDim();
    auto& input_dtype = prev_layer.getDataType();
    auto input_memref = genMemRef(input_shape, input_dtype);
    mlir << input_memref << "\n";

    if(this->isTest()) 
    {
        std::string code = "    %" + std::to_string(input_buffer_reg) 
                         + " = "
                         + dict[ALLOC]
                         + "() : "
                         + input_memref
                         + " \n";
        code += "    %c1 = constant 1.0 : f32 \n";                 
        code += "    " 
                + dict[FILLVAR]
                + "(%c1,%"
                + std::to_string(input_buffer_reg)
                + ") : f32, "
                + input_memref 
                + "\n";
        mlir << code << "\n";

    }

    // Determine output and kernel dimension
    auto &kernel_dim = cur_layer.getKernelDim();
    // assert(input_shape.size() == 1); // Input must be 1D
    mlir << "    // Kernel dim.: ";
    // weight_dim[0] - input dimension
    // weight_dim[1] - output dimension
    for (auto dim : kernel_dim) { mlir << dim << " "; }
    mlir << "\n";
    mlir << "    // Output size: " << kernel_dim[1] << "\n";

    auto &cur_layer_dtype = cur_layer.getDataType();
    auto kernel_name = cur_layer.getName() + "_kernel";
    const auto [it_var1, flag1] = variable_map.insert({kernel_name,global_register_tracker++});
    if(!flag1) {
        std::cout << "Variable map insertion failure" << std::endl;
    }
    auto kernel_reg = variable_map.at(kernel_name);
    auto kernel_memref = genMemRef(kernel_dim, cur_layer_dtype);
    std::string code = "    %" + std::to_string(kernel_reg)
                     + " = "
                     + dict[ALLOC]
                     + "() : "
                     + kernel_memref;
    mlir << code << "\n";

    // alloc output
    std::vector<unsigned> output_shape = {input_shape[0],kernel_dim[1]};
    // for (int i = )
    const auto [it_var2, flag2] = variable_map.insert({cur_layer.getName(),global_register_tracker++});
    if(!flag2) {
        std::cout << "Variable map insertion failure" << std::endl;
    }
    auto output_reg = variable_map.at(cur_layer.getName());
    auto output_memref = genMemRef(output_shape, cur_layer_dtype);
    code = "    %" + std::to_string(output_reg)
                   + " = "
                   + dict[ALLOC]
                   + "() : "
                   + output_memref;
    mlir << code << "\n\n";
    cur_layer.setOutputDim(output_shape);

    // Dense function
    // code = "";
    int num_of_loops = kernel_dim.size();

    // Do Matmul:
    code  = "    " + dict[MATMUL]
          + " ins( %"
          + std::to_string(input_buffer_reg)
          + ", %"
          + std::to_string(kernel_reg)
          + " :"
          + input_memref
          + ", "
          + kernel_memref
          + ") outs(%"
          + std::to_string(output_reg)
          + " :"
          + output_memref
          + " )\n";
    mlir << code;
    mlir << "\n";
}

void MLIRGen::genZeroPadding2DLayer(Layer& prev_layer,
                            Layer& cur_layer)
{
    mlir << "    // Layer type: ZeroPadding2D.\n"
         << "    // Layer name: " << cur_layer.getName() << "\n";

    auto cur_layer_id = cur_layer.getID();
    // auto prev_layer_id = prev_layer.getID();
    auto in_layers = cur_layer.getInLayers();

    if (in_layers.size() == 1) {

        auto prev_layer_id = in_layers[0]->getID(); 
        // mlir << "    // Input from layer: " << prev_layer.getName() << "\n";
        // auto input_buffer_reg = variable_map.at(prev_layer.getName());
        mlir << "    // Input from layer: " << in_layers[0]->getName() << "\n";
        auto input_buffer_reg = variable_map.at(in_layers[0]->getName());
        mlir << "    // Input buffer: %"
             << input_buffer_reg << " : ";
        // auto& input_shape = prev_layer.getOutputDim();
        // auto& input_dtype = prev_layer.getDataType();
        auto& input_shape = in_layers[0]->getOutputDim();
        auto& input_dtype = in_layers[0]->getDataType();
        auto input_memref = genMemRef(input_shape, input_dtype);
        mlir << input_memref << "\n";
        
        if(this->isTest()) 
        {
            std::string code = "    %" + std::to_string(input_buffer_reg) 
                             + " = "
                             + dict[ALLOC]
                             + "() : "
                             + input_memref
                             + " \n";
            code += "    %v1 = constant 1.0 : f32 \n";                 
            code += "    " 
                    + dict[FILLVAR]
                    + "(%v1,%"
                    + std::to_string(input_buffer_reg)
                    + ") : f32, "
                    + input_memref 
                    + "\n";
            mlir << code << "\n";

        }

        auto& output_dtype = cur_layer.getDataType();
        std::vector<unsigned> output_shape;
        
        //TODO:(Vinay) Padding for this layer has 4 dims. Auto detect the sizes?
        // Layer padding
        auto lp = cur_layer.getPaddings();
        output_shape.push_back(input_shape[0]);
        output_shape.push_back(input_shape[1] + 2*lp[0]);
        output_shape.push_back(input_shape[2] + 2*lp[2]);
        output_shape.push_back(input_shape[3]);

        mlir << "    // Output size: ";
        for (auto dim : output_shape) { mlir << dim << " "; }
        mlir << "\n";
        cur_layer.setOutputDim(output_shape);

        // Load input memref to a tensor
        auto &cur_layer_dtype = cur_layer.getDataType();
        auto tensor0_name = cur_layer.getName() + "_tensor0";
        const auto [it_var1, flag1] = variable_map.insert({tensor0_name,global_register_tracker++});
        if(!flag1) {
            std::cout << "Variable map insertion failure" << std::endl;
        }
        auto tensor0_reg = variable_map.at(tensor0_name);
        auto tensor0_memref = genMemRef(input_shape, cur_layer_dtype);
        auto tensorOut_memref = genMemRef(output_shape, cur_layer_dtype);
        // auto tensor0_memref = genMemRef2(input_shape, cur_layer_dtype);
        // auto tensorOut_memref = genMemRef2(output_shape, cur_layer_dtype);

        std::string code = "    %" + std::to_string(tensor0_reg)
                            + " = "
                            + "memref.tensor_load %"
                            + std::to_string(input_buffer_reg)
                            + " : "
                            + tensor0_memref
                            // + ", #layout, memspace0>";
                            + "\n";
        mlir << code;

        // Pad tensor
        // TODO: (Vinay), (automate) handle constant variables
        // TODO: (Vinay), layer data type, int32/f32
        auto tensor0_tensor = genTensor(input_shape, cur_layer_dtype);
        auto tensor1_tensor = genTensor(output_shape, cur_layer_dtype);
        code = "    %v0 = constant 0.0 : f32 \n";
        mlir << code;
        // code = "    %pad_value = %const0 : f32";
        // mlir << code << "\n";
        std::string tensor_store = "%tensor_store";
        // code = "    %" + std::to_string(tensor0_reg)
        code = "    " + tensor_store
                + " = "
                + "linalg.pad_tensor %"
                + std::to_string(tensor0_reg)
                + " low[0," + std::to_string(lp[0])
                + ", " + std::to_string(lp[1]) + ",0]"
                + " high[0," + std::to_string(lp[2])
                + ", " + std::to_string(lp[3]) + ",0] {\n"
                + "    ^bb0(%arg0 : index, %arg1 : index, %arg2 : index, %arg3 : index):\n"
                + "      linalg.yield %v0 : f32\n"
                + "    } : "
                + tensor0_tensor
                + " to "
                + tensor1_tensor
                + "\n";
        mlir << code;

        // TODO: (Vinay) Free above memref? 
        // Store Tensor to Memref
        // Get Dim:
        
        // auto tensor0Dim_name =  cur_layer.getName() + "_tensor0Dim";
        // const auto [it_var1, flag2] = variable_map.insert({tensor0Dim_name,global_register_tracker++});
        // if(!flag2) {
        //     std::cout << "Variable map insertion failure" << std::endl;
        // }
        // auto tensor0Dim_reg = variable_map.at(tensor0_name);
        // code = "    %" + std::to_string(tensor0Dim_reg) 
        //         + " dim %" + std::to_string(tensor0_reg)

        // Output SSA
        const auto [it_var2, flag2] = variable_map.insert({cur_layer.getName(),global_register_tracker++});
        if(!flag2) {
            std::cout << "Variable map insertion failure" << std::endl;
        }
        auto output_reg = variable_map.at(cur_layer.getName());
        code = "    %" + std::to_string(output_reg) 
                + " = " 
                + dict[ALLOC]
                + "() : "
                + tensorOut_memref //+ ", #layout, memspace0>" 
                + "\n";

        code += "    memref.tensor_store "
                // + std::to_string(tensor0_reg)
                + tensor_store
                + ", %"
                + std::to_string(output_reg)
                + " : " 
                + tensorOut_memref //+ ", #layout, memspace0>"
                + "\n"
                ;
        mlir << code; 
        mlir << "\n";

    }
}

void MLIRGen::genBatchNormalizationLayer(Layer& prev_layer,
                            Layer& cur_layer)
{
    mlir << "    // Layer type: BatchNormalization (Conv2D)\n"
         << "    // Layer name: " << cur_layer.getName() << "\n";
    auto prev_layer_id = prev_layer.getID();
    auto cur_layer_id = cur_layer.getID();
    mlir << "    // Input from layer: " << prev_layer.getName() << "\n";
    auto input_buffer_reg = variable_map.at(prev_layer.getName());
    mlir << "    // Input buffer: %"
         << input_buffer_reg << " : ";
    auto& input_shape = prev_layer.getOutputDim();
    auto& input_dtype = prev_layer.getDataType();
    auto input_memref = genMemRef(input_shape, input_dtype);
    mlir << input_memref << "\n";

    if(this->isTest()) 
    {
        std::string code = "    %" + std::to_string(input_buffer_reg) 
                         + " = "
                         + dict[ALLOC]
                         + "() : "
                         + input_memref
                         + " \n";
        code += "    %v1 = constant 1.0 : f32 \n";                 
        code += "    " 
                + dict[FILLVAR]
                + "(%v1,%"
                + std::to_string(input_buffer_reg)
                + ") : f32, "
                + input_memref 
                + "\n";
        mlir << code << "\n";

    }

    // Output Shape, Dim 
    auto& output_dtype = cur_layer.getDataType();
    auto &cur_layer_dtype = cur_layer.getDataType();
    std::vector<unsigned> output_shape = input_shape;
    mlir << "    // Output size: ";
    for (auto dim : output_shape) { mlir << dim << " "; }
    mlir << "\n";
    cur_layer.setOutputDim(output_shape);

    // alloc output 
    const auto [it_var1, flag1] = variable_map.insert({cur_layer.getName(),global_register_tracker++});
    if(!flag1) {
        std::cout << "Variable map insertion failure" << std::endl;
    }
    auto output_reg = variable_map.at(cur_layer.getName());
    auto output_memref = genMemRef(output_shape, cur_layer_dtype);

    // If input_shape[0] > 1 i.e. batch size is greater than 1, apply batch norm, else none. 
    // TODO: If epsilon variable can be read, we can get rid of this condition. 
    if(input_shape[0] > 0) {
        // alloc memory for sum
        
        // auto batch_sum_name = cur_layer.getName() + "_batch_sum";
        // std::vector<unsigned> batch_sum_shape = input_shape; 
        // batch_sum_shape[0] = 1; // adding along the batch axis
        // const auto [it_var2, flag2] = variable_map.insert({batch_sum_name,global_register_tracker++});
        // if(!flag2) {
        //     std::cout << "Variable map insertion failure" << std::endl;
        // }
        // auto batch_sum_reg = variable_map.at(batch_sum_name);
        // auto batch_sum_memref = genMemRef(batch_sum_shape, cur_layer_dtype);
        
        // Get Mean 
        auto moving_mean_dims = cur_layer.getMovingMeanDim();    
        auto code = "    // Moving Mean dimensions -- "
                        + std::to_string(moving_mean_dims.size());
        mlir << code << "\n";

        auto moving_mean_val = cur_layer.getMovingMeanData();    
        code = "    // Moving Mean Val -- "
                        + std::to_string(moving_mean_val[0]);
        mlir << code << "\n";

        // Get Variance 
        auto moving_var_dims = cur_layer.getMovingVarianceDim();    
        code = "    // Moving Variance dimensions -- "
                        + std::to_string(moving_var_dims.size());
        mlir << code << "\n";
        
        auto moving_variance_val = cur_layer.getMovingVarianceData();    
        code = "    // Moving Variance Val -- "
                        + std::to_string(moving_variance_val[0]);
        mlir << code << "\n";

        // Get Epsilon Value if present. 
        code = "    // Epsilon value -- "
                    + std::to_string(cur_layer.getEpsilon());   

        mlir << code << "\n";
        
        
        auto gamma_vec = cur_layer.getGamma();
        code = "    // Gamma len -- "
                    + std::to_string(gamma_vec.size());
        mlir << code << "\n";   

        auto beta_vec = cur_layer.getBeta();
        code = "    // Beta len -- "
                    + std::to_string(beta_vec.size());
        mlir << code << "\n";
        
        code = "    %" + std::to_string(output_reg)
                       + " = "
                       + dict[ALLOC]
                       + "() : "
                       + output_memref;
        mlir << code << "\n";
        
        // Load gamma and beta variables
        auto gamma_name = cur_layer.getName() + "_gamma";
        auto gamma_dim = cur_layer.getGammaDim();
        const auto [it_var1a, flag1a] = variable_map.insert({gamma_name,global_register_tracker++});
        if(!flag1a) {
            std::cout << "Variable map insertion failure" << std::endl;
        }
        auto gamma_reg = variable_map.at(gamma_name);
        auto gamma_memref = genMemRef(gamma_dim, cur_layer_dtype);
        code = "    %" + std::to_string(gamma_reg)
                         + " = memref.get_global @"
                         + gamma_name
                         + " : "
                         + gamma_memref;
        mlir << code << "\n";

        auto beta_name = cur_layer.getName() + "_beta";
        auto beta_dim = cur_layer.getBetaDim();
        const auto [it_var1b, flag1b] = variable_map.insert({beta_name,global_register_tracker++});
        if(!flag1b) {
            std::cout << "Variable map insertion failure" << std::endl;
        }
        auto beta_reg = variable_map.at(beta_name);
        auto beta_memref = genMemRef(beta_dim, cur_layer_dtype);
        code = "    %" + std::to_string(beta_reg)
                         + " = memref.get_global @"
                         + beta_name
                         + " : "
                         + beta_memref;
        mlir << code << "\n";

        // -------------------------------------
        std::string index_ssa =  "    %zero = constant 0.0 : f32 \n";
        index_ssa +=  "    %ci0 = constant 0 : index \n";
        index_ssa +=  "    %ci1 = constant 1 : index \n";
        std::unordered_map<int, int> input_index_map;         
        for (int i = 0; i < input_shape.size(); i++) 
        {
            // const auto [it]
            input_index_map.insert({input_shape[i], i});
        }

        for ( auto i : input_index_map)
        {
            index_ssa +=  "    %ci_Shape_" + std::to_string(i.first) + " = "+ " constant " + std::to_string(i.first) + " : index \n";
        }

        mlir << index_ssa << "\n";

        code = "    %c0 = constant 0 : index \n";
        code += "    %c1 = constant 1 : index \n";
        // Gen Loop for Sum 
        int num_loops = input_shape.size();
        // inner 3 loops first
        for (auto i = 0; i < num_loops; i++)
        {
            std::string one_loop =
                std::string(4 + i * 2, ' ')
                + dict[FOR]
                + " %" + default_index_str[i] + " = %ci0 to "
                + "%ci_Shape_"+ std::to_string(input_shape[i])
                + " step %ci1\n"
                + std::string(4 + (i) * 2, ' ') + "{\n";
                // + std::string(4 + (i-1) * 2, ' ') + "{\n";
            code += one_loop;
        }
        
        // --------------------------------------------------------------
        // 1D array for batch sum idx[i] 
        auto batch_sum_1d_name = cur_layer.getName() + "_batch_1d_sum";
        std::vector<unsigned> batch_sum_1d_shape (1,input_shape[0]); 
        auto batch_sum_1d_memref = genMemRef(batch_sum_1d_shape, cur_layer_dtype);
        // batch_sum_shape[0] = 1; // adding along the batch axis
        const auto [it_var3, flag3] = variable_map.insert({batch_sum_1d_name,global_register_tracker++});
        if(!flag3) {
            std::cout << "Variable map insertion failure" << std::endl;
        }
        auto batch_sum_1d_reg = variable_map.at(batch_sum_1d_name);
        
        // code += std::string(4 + (num_loops-1) * 2, ' ')
        //        + "%" + std::to_string(batch_sum_1d_reg)
        //        + " = "
        //        + dict[ALLOC]
        //        + "() : "
        //        + batch_sum_1d_memref
        //        + "\n";

        code += std::string(4 + (num_loops) * 2, ' ')  
                + " // Load Mean for Batch Norm\n";
        std::string mean_bn = "%mean_" + cur_layer.getName();
        code += std::string(4 + (num_loops) * 2, ' ') 
                + mean_bn 
                + " = constant "
                + std::to_string(moving_mean_val[0]) 
                + " : "
                + genDataType(cur_layer_dtype)
                + "\n";


        // code += std::string(4 + (num_loops-1) * 2, ' ') + " // Evaluate Mean\n";
        // //outermost loop 
        // std::string loop_open =
        //     std::string(4 + (num_loops-1) * 2, ' ')
        //     + dict[FOR]
        //     + " %" + default_index_str[0] + " = 0 to "
        //     + std::to_string(input_shape[0])
        //     + " step 1\n"
        //     + std::string(4 + (num_loops-1) * 2, ' ') + "{\n";
        // code += loop_open;

        // //load data to temp variable 
        // // TODO: (Vinay) Separate Variable Tracking for temp variables
        // auto load_t0_name = cur_layer.getName() + "_load_t0";
        // const auto [it_var4, flag4] = variable_map.insert({load_t0_name,global_register_tracker++});
        // if(!flag4) {
        //     std::cout << "Variable map insertion failure" << std::endl;
        // }
        // auto load_t0_reg = variable_map.at(load_t0_name);
        // std::string tmp_load_str = 
        //     std::string(4 + (num_loops)*2, ' ')
        //     + "%" + std::to_string(load_t0_reg) 
        //     + " = " 
        //     + genLoad(default_index_str,
        //                 input_buffer_reg,
        //                 0,
        //                 input_shape.size(),
        //                 input_memref) + "\n";
        // code += tmp_load_str;
        // std::string store_t0_str = "%" + std::to_string(load_t0_reg);
        // std::string tmp_store_str = 
        //     std::string(4 + (num_loops)*2, ' ')
        //     + genStore(default_index_str,
        //                        store_t0_str,
        //                        batch_sum_1d_reg,
        //                        0,
        //                        1,
        //                        batch_sum_1d_memref)
        //             + "\n"; 
        // code += tmp_store_str;

        // std::string loop_close = std::string(4 + (num_loops - 1)*2, ' ') + "}\n";
        // code += loop_close;

        // Temp Variable for 1D Sum
        // auto sum_t0_name = cur_layer.getName() + "_sum_t0";
        // const auto [it_var5, flag5] = variable_map.insert({sum_t0_name,global_register_tracker++});
        // if(!flag5) {
        //     std::cout << "Variable map insertion failure" << std::endl;
        // }
        // auto sum_t0_reg = variable_map.at(sum_t0_name);
        // // auto sum_t0_memref = genMemRef(batch_sum_1d_shape, cur_layer_dtype); // single element

        // // Get Sum1D
        // std::string sum_var = "%sum";
        // unsigned space_scaling = num_loops - 1;
        // code += genSum1D(sum_var, batch_sum_1d_shape[0], batch_sum_1d_reg, 
        //                             batch_sum_1d_memref, cur_layer_dtype, space_scaling);

        // // Get Sum Length
        // std::string arr_length = "%arr_len";
        // code += std::string(4 + (num_loops - 1)*2, ' ') 
        //         + arr_length
        //         + " = constant "
        //         + std::to_string(batch_sum_1d_shape[0])
        //         + " : "
        //         + genDataType(cur_layer_dtype)
        //         + "\n";

        // // Get Mean for Arr: 
        // std::string mean_var = "%mean";
        // code += std::string(4 + (num_loops - 1)*2, ' ') 
        //         + mean_var
        //         + " = "
        //         + dict[DIVF]
        //         + " "
        //         + arr_length
        //         + " , "
        //         + sum_var
        //         + " : "
        //         + genDataType(cur_layer_dtype)
        //         + "\n";

        // --------------------------------------------------------------
        // 1D array for batch var_sum idx[i]
        // auto batch_var_sum_1d_name = cur_layer.getName() + "_batch_1d_var_sum";
        // std::vector<unsigned> batch_var_sum_1d_shape (1,input_shape[0]); 
        // // batch_sum_shape[0] = 1; // adding along the batch axis
        // const auto [it_var6, flag6] = variable_map.insert({batch_var_sum_1d_name,global_register_tracker++});
        // if(!flag6) {
        //     std::cout << "Variable map insertion failure" << std::endl;
        // }
        // auto batch_var_sum_1d_reg = variable_map.at(batch_var_sum_1d_name);
        // auto batch_var_sum_1d_memref = genMemRef(batch_var_sum_1d_shape, cur_layer_dtype);
        // code += std::string(4 + (num_loops-1) * 2, ' ')
        //        + "%" + std::to_string(batch_var_sum_1d_reg)
        //        + " = "
        //        + dict[ALLOC]
        //        + "() : "
        //        + batch_var_sum_1d_memref
        //        + "\n";

        // code += std::string(4 + (num_loops-1) * 2, ' ') + " // Evaluate Variance\n";
        // code += std::string(4 + (num_loops-1) * 2, ' ') + " // (x[i] - u)^2 \n";
        // std::string loop_open_var =
        //     std::string(4 + (num_loops-1) * 2, ' ')
        //     + dict[FOR]
        //     + " %" + default_index_str[0] + " = 0 to "
        //     + std::to_string(input_shape[0])
        //     + " step 1\n"
        //     + std::string(4 + (num_loops-1) * 2, ' ') + "{\n";
        // code += loop_open_var;

        // //load data to temp variable 
        // auto load_t1_name = cur_layer.getName() + "_load_t1";
        // const auto [it_var7, flag7] = variable_map.insert({load_t1_name,global_register_tracker++});
        // if(!flag7) {
        //     std::cout << "Variable map insertion failure" << std::endl;
        // }
        // auto load_t1_reg = variable_map.at(load_t1_name);
        // tmp_load_str = std::string(4 + (num_loops)*2, ' ')
        //                 + "%" + std::to_string(load_t1_reg) 
        //                 + " = " 
        //                 + genLoad(default_index_str,
        //                             batch_var_sum_1d_reg,
        //                             0,
        //                             batch_var_sum_1d_shape.size(),
        //                             batch_var_sum_1d_memref) + "\n";
        // code += tmp_load_str;
        
        // // (x[i] - u)^2
        // auto diff_t0_name = cur_layer.getName() + "_diff_t0";
        // const auto [it_var8, flag8] = variable_map.insert({diff_t0_name,global_register_tracker++});
        // if(!flag8) {
        //     std::cout << "Variable map insertion failure" << std::endl;
        // }
        // auto diff_t0_reg = variable_map.at(diff_t0_name);
        // code += std::string(4 + (num_loops)*2, ' ') 
        //         + "%" + std::to_string(diff_t0_reg) 
        //         + " = "
        //         + dict[SUBF]
        //         + " %"
        //         + std::to_string(load_t1_reg)
        //         + ", "
        //         + mean_var
        //         + " : "
        //         + genDataType(cur_layer_dtype)
        //         + "\n";
        
        // auto sq_t0_name = cur_layer.getName() + "_sq_t0";
        // const auto [it_var9, flag9] = variable_map.insert({sq_t0_name,global_register_tracker++});
        // if(!flag9) {
        //     std::cout << "Variable map insertion failure" << std::endl;
        // }
        // auto sq_t0_reg = variable_map.at(sq_t0_name);
        // // TODO: (Vinay) square, cube
        // code += std::string(4 + (num_loops)*2, ' ')
        //         + "%" + std::to_string(sq_t0_reg)
        //         + " = "
        //         + dict[MULF] 
        //         + " %"
        //         + std::to_string(diff_t0_reg)
        //         + ", %"
        //         + std::to_string(diff_t0_reg)
        //         + " : "
        //         + genDataType(cur_layer_dtype)
        //         + "\n";
        // // Store
        // std::string store_t1_str = "%" + std::to_string(sq_t0_reg);
        // code += std::string(4 + (num_loops)*2, ' ')
        //         + genStore(default_index_str,
        //                        store_t1_str,
        //                        batch_var_sum_1d_reg,
        //                        0,
        //                        1,
        //                        batch_var_sum_1d_memref)
        //         + "\n"; 
        // loop_close = std::string(4 + (num_loops - 1)*2, ' ') + "}\n";
        // code += loop_close;
        
        // // Gen Sum1D
        // sum_var = "%sumVar";
        // space_scaling = num_loops - 1;
        // code += genSum1D(sum_var, batch_var_sum_1d_shape[0], batch_var_sum_1d_reg, 
        //                             batch_var_sum_1d_memref, cur_layer_dtype, space_scaling);
        // // Gen Var
        // std::string variance_var = "%var";
        // code += std::string(4 + (num_loops - 1)*2, ' ') 
        //         + variance_var
        //         + " = "
        //         + dict[DIVF]
        //         + " "
        //         + arr_length
        //         + " , "
        //         + sum_var
        //         + " : "
        //         + genDataType(cur_layer_dtype)
        //         + "\n";

        // --------------------------------------------------------------
        code += std::string(4 + (num_loops) * 2, ' ') 
                + " // Load Variance     for Batch Norm\n";
        std::string variance_bn = "%variance_" + cur_layer.getName();
        code += std::string(4 + (num_loops) * 2, ' ') 
                + variance_bn 
                + " = constant "
                + std::to_string(moving_variance_val[0]) 
                + " : "
                + genDataType(cur_layer_dtype)
                + "\n";

        // Var - epsilon
        code += std::string(4 + (num_loops) * 2, ' ') 
                + " // Load Epsilon for Batch Norm\n";
        std::string epsilon_bn = "%epsilon_" + cur_layer.getName();
        code += std::string(4 + (num_loops) * 2, ' ') 
                + epsilon_bn 
                + " = constant "
                + std::to_string(cur_layer.getEpsilon())
                + " : "
                + genDataType(cur_layer_dtype)
                + "\n";
        
        // std::string variance_sub_ep_var = "%var_sub_ep";

        // Evaluate for each array element and store
        
        // //Open loop 
        // std::string loop_open =
        //     std::string(4 + (num_loops-1) * 2, ' ')
        //     + dict[FOR]
        //     + " %" + default_index_str[0] + " = 0 to "
        //     + std::to_string(input_shape[0])
        //     + " step 1\n"
        //     " step 1\n"
        // code += loop_open;

        //load data to temp variable
        auto load_t2_name = cur_layer.getName() + "_load_t2";
        const auto [it_var10, flag10] = variable_map.insert({load_t2_name,global_register_tracker++});
        if(!flag10) {
            std::cout << "Variable map insertion failure" << std::endl;
        }
        auto load_t2_reg = variable_map.at(load_t2_name);
        std::string tmp_load_str = std::string(4 + (num_loops)*2, ' ')
                        + "%" + std::to_string(load_t2_reg)
                        + " = " 
                        + genLoad(default_index_str,
                                    // batch_sum_1d_reg,
                                    input_buffer_reg, 
                                    0,
                                    // batch_sum_1d_shape.size(),
                                    input_shape.size(),
                                    input_memref) + "\n";
                                    // batch_sum_1d_memref) + "\n";
        code += tmp_load_str;

        auto diff_t1_name = cur_layer.getName() + "_diff_t1";
        const auto [it_var11, flag11] = variable_map.insert({diff_t1_name,global_register_tracker++});
        if(!flag11) {
            std::cout << "Variable map insertion failure" << std::endl;
        }
        auto diff_t1_reg = variable_map.at(diff_t1_name);
        code += std::string(4 + (num_loops)*2, ' ') 
                + "%" + std::to_string(diff_t1_reg) 
                + " = "
                + dict[SUBF]
                + " %"
                + std::to_string(load_t2_reg)
                + ", "
                + mean_bn
                + " : "
                + genDataType(cur_layer_dtype)
                + "\n";

        // Sqrt Variance and Epsilon : Latter not implemented
        // TODO: Implement for epsilon
        auto diff_t2_name = cur_layer.getName() + "_diff_t2";
        const auto [it_var12, flag12] = variable_map.insert({diff_t2_name,global_register_tracker++});
        if(!flag12) {
            std::cout << "Variable map insertion failure" << std::endl;
        }
        auto diff_t2_reg = variable_map.at(diff_t2_name);
        code += std::string(4 + (num_loops)*2, ' ') 
                + "%" + std::to_string(diff_t2_reg) 
                + " = "
                + dict[ADDF]
                // + std::to_string(load_t2_reg)
                + " "
                + epsilon_bn
                + ", "
                + variance_bn
                + " : "
                + genDataType(cur_layer_dtype)
                + "\n";

        auto rsqrt_t0_name = cur_layer.getName() + "_rsqrt_t0";
        const auto [it_var13, flag13] = variable_map.insert({rsqrt_t0_name,global_register_tracker++});
        if(!flag13) {
            std::cout << "Variable map insertion failure" << std::endl;
        }
        auto rsqrt_t0_reg = variable_map.at(rsqrt_t0_name);
        code += std::string(4 + (num_loops)*2, ' ')
                + "%" + std::to_string(rsqrt_t0_reg)
                + " = "
                + dict[RSQRT]
                + " %"+ std::to_string(diff_t2_reg)
                + " : "
                + genDataType(cur_layer_dtype)
                + "\n";
        
        auto mult_t0_name = cur_layer.getName() + "_mult_t0";
        const auto [it_var14, flag14] = variable_map.insert({mult_t0_name,global_register_tracker++});
        if(!flag13) {
            std::cout << "Variable map insertion failure" << std::endl;
        }
        auto mult_t0_reg = variable_map.at(mult_t0_name);
        code += std::string(4 + (num_loops)*2, ' ')
                + "%" + std::to_string(mult_t0_reg)
                + " = "
                + dict[MULF]
                + " %"
                + std::to_string(rsqrt_t0_reg)
                + ", %"
                + std::to_string(diff_t1_reg)
                + " : "
                + genDataType(cur_layer_dtype)
                + "\n";
        
        // TODO Scale and Shift. 
        // Load Gamma variable: 
        auto load_t3_name = cur_layer.getName() + "_load_t3";
        const auto [it_var15, flag15] = variable_map.insert({load_t3_name,global_register_tracker++});
        if(!flag15) {
            std::cout << "Variable map insertion failure" << std::endl;
        }
        auto load_t3_reg = variable_map.at(load_t3_name);
        tmp_load_str = std::string(4 + (num_loops)*2, ' ')
                        + "%" + std::to_string(load_t3_reg)
                        + " = " 
                        + genLoad(default_index_str,
                                    // batch_sum_1d_reg,
                                    gamma_reg, 
                                    3,
                                    // batch_sum_1d_shape.size(),
                                    input_shape.size(),
                                    gamma_memref) + "\n";
                                    // batch_sum_1d_memref) + "\n";
        code += tmp_load_str;

        // Load Beta variable:
        auto load_t4_name = cur_layer.getName() + "_load_t4";
        const auto [it_var16, flag16] = variable_map.insert({load_t4_name,global_register_tracker++});
        if(!flag16) {
            std::cout << "Variable map insertion failure" << std::endl;
        }
        auto load_t4_reg = variable_map.at(load_t4_name);
        tmp_load_str = std::string(4 + (num_loops)*2, ' ')
                        + "%" + std::to_string(load_t4_reg)
                        + " = " 
                        + genLoad(default_index_str,
                                    // batch_sum_1d_reg,
                                    beta_reg, 
                                    3,
                                    // batch_sum_1d_shape.size(),
                                    input_shape.size(),
                                    beta_memref) + "\n";
                                    // batch_sum_1d_memref) + "\n";
        code += tmp_load_str;
        // Apply scale: 
        auto mult_t1_name = cur_layer.getName() + "_mult_t1";
        const auto [it_var17, flag17] = variable_map.insert({mult_t1_name,global_register_tracker++});
        if(!flag17) {
            std::cout << "Variable map insertion failure" << std::endl;
        }
        auto mult_t1_reg = variable_map.at(mult_t1_name);
        code += std::string(4 + (num_loops)*2, ' ')
                + "%" + std::to_string(mult_t1_reg)
                + " = "
                + dict[MULF]
                + " %"
                + std::to_string(load_t3_reg)
                + ", %"
                + std::to_string(mult_t0_reg)
                + " : "
                + genDataType(cur_layer_dtype)
                + "\n";
        
        // Apply shift: 
        auto add_t0_name = cur_layer.getName() + "_add_t0";
        const auto [it_var18, flag18] = variable_map.insert({add_t0_name,global_register_tracker++});
        if(!flag18) {
            std::cout << "Variable map insertion failure" << std::endl;
        }
        auto add_t0_reg = variable_map.at(add_t0_name);
        code += std::string(4 + (num_loops)*2, ' ')
                + "%" + std::to_string(add_t0_reg)
                + " = "
                + dict[ADDF]
                + " %"
                + std::to_string(load_t4_reg)
                + ", %"
                + std::to_string(mult_t1_reg)
                + " : "
                + genDataType(cur_layer_dtype)
                + "\n";

        // Store result 
        code += std::string(4 + (num_loops-1) * 2, ' ') + " // Store Normalized Value\n";
        std::string store_t2_str = "%" + std::to_string(add_t0_reg);
        // IDX vector 
        std::vector<unsigned> dflt_idx_vector_seq;
        for (int i=0; i < output_shape.size(); i++)
            dflt_idx_vector_seq.push_back(i);
        code += std::string(4 + (num_loops)*2, ' ')
                + genStore(default_index_str,
                           store_t2_str,
                           output_reg,
                           dflt_idx_vector_seq,
                           output_memref)
                + "\n";
       
        for (int i = output_shape.size() - 1; i >= 0; i--)
        {
            std::string loop_nest =
                std::string(4 + i * 2, ' ') + "}\n";
            code += loop_nest;
        }
        code += "\n";
        mlir << code << "\n";
    }
    else
    {
        auto code = "    %" + std::to_string(output_reg) 
                    + " = "
                    + "%" + std::to_string(input_buffer_reg);
                    // + " : "
                    // + output_memref;
        mlir << code << "\n";

    }
}

void MLIRGen::genAddLayer(Layer& prev_layer,
                            Layer& cur_layer)
{
    mlir << "    // Layer type: Add Layer.\n"
         << "    // Layer name: " << cur_layer.getName() << "\n";

    auto cur_layer_id = cur_layer.getID();
    auto in_layers = cur_layer.getInLayers();
    // Current Add Layers can handle only two inputs
    auto prev_layer_id_0 = in_layers[0]->getID();
    auto prev_layer_id_1 = in_layers[1]->getID();
    mlir << "    // Input from layer: " << in_layers[0]->getName() << "\n";
    auto input_buffer_reg_0 = variable_map.at(in_layers[0]->getName());
    mlir << "    // Input buffer: %"
         << input_buffer_reg_0 << " : ";
    auto& input_shape = in_layers[0]->getOutputDim();
    auto& input_shape_1 = in_layers[1]->getOutputDim();
    assert(input_shape_1 == input_shape);
    auto& input_dtype = in_layers[0]->getDataType();
    auto input_memref = genMemRef(input_shape, input_dtype);
    mlir << input_memref << "\n";
  
    mlir << "    // Input from layer: " << in_layers[1]->getName() << "\n";
        auto input_buffer_reg_1 = variable_map.at(in_layers[1]->getName());
            mlir << "    // Input buffer: %"
             << input_buffer_reg_1 << " : ";
    mlir << input_memref << "\n";

    if(this->isTest()) 
    {
        std::string code = "    %" + std::to_string(input_buffer_reg_0) 
                 + " = "
                 + dict[ALLOC]
                 + "() : "
                 + input_memref
                 + " \n";
        code += "    %" + std::to_string(input_buffer_reg_1) 
                 + " = "
                 + dict[ALLOC]
                 + "() : "
                 + input_memref
                 + " \n";

        code += "    %v1 = constant 1.0 : f32 \n";                 
        code += "    " 
                + dict[FILLVAR]
                + "(%v1,%"
                + std::to_string(input_buffer_reg_0)
                + ") : f32, "
                + input_memref 
                + "\n";
        code += "    " 
                + dict[FILLVAR]
                + "(%v1,%"
                + std::to_string(input_buffer_reg_1)
                + ") : f32, "
                + input_memref 
                + "\n";
        mlir << code << "\n";

    }

    auto& output_dtype = cur_layer.getDataType();
    auto output_shape = input_shape;
    // std::vector<unsigned> output_shape = input_shape;
    // output_shape.push_back(input_shape[0]);


    mlir << "    // Output size: ";
        for (auto dim : output_shape) { mlir << dim << " "; }
    mlir << "\n";
    cur_layer.setOutputDim(output_shape);
    auto &cur_layer_dtype = cur_layer.getDataType();
    const auto [it_var0, flag0] = variable_map.insert({cur_layer.getName(),global_register_tracker++});
    if(!flag0) {
        std::cout << "Variable map insertion failure" << std::endl;
    }
    auto output_reg = variable_map.at(cur_layer.getName());
    auto output_memref = genMemRef(output_shape, cur_layer_dtype);
    // alloc output_reg 
    std::string code = "    %" + std::to_string(output_reg)
                       + " = "
                       + dict[ALLOC]
                       + "() : "
                       + output_memref
                       + "\n";
    mlir << code;
    // -------------------------------------
    std::string index_ssa =  "    %zero = constant 0.0 : f32 \n";
    index_ssa +=  "    %ci0 = constant 0 : index \n";
    index_ssa +=  "    %ci1 = constant 1 : index \n";
    std::unordered_map<int, int> input_index_map;         
    for (int i = 0; i < input_shape.size(); i++) 
    {
        // const auto [it]
        input_index_map.insert({input_shape[i], i});
    }

    for ( auto i : input_index_map)
    {
        index_ssa +=  "    %ci_Shape_" + std::to_string(i.first) + " = "+ " constant " + std::to_string(i.first) + " : index \n";
    }

    mlir << index_ssa << "\n";

    code = "";
    // Gen Loop
    int num_loops = input_shape.size();
    for (auto i = 0; i < num_loops; i++)
    {
        std::string one_loop =
            std::string(4 + i * 2, ' ')
            + dict[FOR]
            + " %" + default_index_str[i] + " = %ci0 to "
            + "%ci_Shape_"+ std::to_string(input_shape[i])
            + " step %ci1\n"
            + std::string(4 + (i) * 2, ' ') + "{\n";
            // + std::string(4 + (i-1) * 2, ' ') + "{\n";
        code += one_loop;
    }

    // load temp data
    auto load_t0_name = cur_layer.getName() + "_load_t0";
    const auto [it_var1, flag1] = variable_map.insert({load_t0_name,global_register_tracker++});
    if(!flag1) {
        std::cout << "Variable map insertion failure" << std::endl;
    }
    auto load_t0_reg = variable_map.at(load_t0_name);
    code += std::string(4 + (num_loops)*2, ' ')
                        + "%" + std::to_string(load_t0_reg)
                        + " = " 
                        + genLoad(default_index_str,
                                    input_buffer_reg_0, 
                                    0,
                                    input_shape.size(),
                                    input_memref) + "\n";
    auto load_t1_name = cur_layer.getName() + "_load_t1";
    const auto [it_var2, flag2] = variable_map.insert({load_t1_name,global_register_tracker++});
    if(!flag2) {
        std::cout << "Variable map insertion failure" << std::endl;
    }
    auto load_t1_reg = variable_map.at(load_t1_name);
    code += std::string(4 + (num_loops)*2, ' ')
                        + "%" + std::to_string(load_t1_reg)
                        + " = " 
                        + genLoad(default_index_str,
                                    input_buffer_reg_1, 
                                    0,
                                    input_shape.size(),
                                    input_memref) + "\n";
    

    auto add_t0_name = cur_layer.getName() + "_add_t0";
    const auto [it_var3, flag3] = variable_map.insert({add_t0_name,global_register_tracker++});
    if(!flag3) {
        std::cout << "Variable map insertion failure" << std::endl;
    }
    auto add_t0_reg = variable_map.at(add_t0_name);
    code += std::string(4 + (num_loops)*2, ' ') 
            + "%" + std::to_string(add_t0_reg) 
            + " = "
            + dict[ADDF]
            // + std::to_string(load_t2_reg)
            + " %"
            + std::to_string(load_t0_reg)
            + ", %"
            + std::to_string(load_t1_reg)
            + " : "
            + genDataType(cur_layer_dtype)
            + "\n";

    // Store result 
    code += std::string(4 + (num_loops) * 2, ' ') + " // Store Sum Value\n";
    std::string store_t2_str = "%" + std::to_string(add_t0_reg);
    // IDX vector 
    std::vector<unsigned> dflt_idx_vector_seq;
    for (int i=0; i < output_shape.size(); i++)
        dflt_idx_vector_seq.push_back(i);
    code += std::string(4 + (num_loops)*2, ' ')
            + genStore(default_index_str,
                       store_t2_str,
                       output_reg,
                       dflt_idx_vector_seq,
                       output_memref)
            + "\n";

    for (int i = output_shape.size() - 1; i >= 0; i--)
    {
        std::string loop_nest =
            std::string(4 + i * 2, ' ') + "}\n";
        code += loop_nest;
    }
    code += "\n";
    mlir << code;



}

void MLIRGen::genGlobalAveragePooling2DLayer(Layer& prev_layer,
                            Layer& cur_layer)
{
    // mlir << " GlobalAveragePooling2D _todo.\n";
    mlir << "    // Layer type: GlobalAveragePooling2D Layer.\n"
         << "    // Layer name: " << cur_layer.getName() << "\n";
    
    auto cur_layer_id = cur_layer.getID();
    auto in_layers = cur_layer.getInLayers();
    // Single input: 
    auto prev_layer_id_0 = in_layers[0]->getID();
    mlir << "    // Input from layer: " << in_layers[0]->getName() << "\n";
    auto input_buffer_reg_0 = variable_map.at(in_layers[0]->getName());
    mlir << "    // Input buffer: %"
         << input_buffer_reg_0 << " : ";
    auto& input_shape = in_layers[0]->getOutputDim();
    auto& input_dtype = in_layers[0]->getDataType();
    auto input_memref = genMemRef(input_shape, input_dtype);
    mlir << input_memref << "\n";
    
    if(this->isTest()) 
    {
        std::string code = "    %" + std::to_string(input_buffer_reg_0) 
                         + " = "
                         + dict[ALLOC]
                         + "() : "
                         + input_memref
                         + " \n";
        code += "    %v1 = constant 1.0 : f32 \n";                 
        code += "    " 
                + dict[FILLVAR]
                + "(%v1,%"
                + std::to_string(input_buffer_reg_0)
                + ") : f32, "
                + input_memref 
                + "\n";
        mlir << code << "\n";

    }

    int num_loops = input_shape.size();
    // Output Shape, Dim 
    auto& output_dtype = cur_layer.getDataType();
    std::vector<unsigned> output_shape; 
    output_shape.push_back(input_shape[0]);
    output_shape.push_back(input_shape[3]);
    auto &cur_layer_dtype = cur_layer.getDataType();
    mlir << "    // Output size: ";
    for (auto dim : output_shape) { mlir << dim << " "; }
    mlir << "\n";
    cur_layer.setOutputDim(output_shape);

    // alloc output 
    const auto [it_var1, flag1] = variable_map.insert({cur_layer.getName(),global_register_tracker++});
    if(!flag1) {
        std::cout << "Variable map insertion failure" << std::endl;
    }
    auto output_reg = variable_map.at(cur_layer.getName());
    auto output_memref = genMemRef(output_shape, cur_layer_dtype);
    std::string code = "    %" + std::to_string(output_reg) 
                       + " = "
                       + dict[ALLOC]
                       + "() : "
                       + output_memref;
    mlir << code << "\n";


    // Assuming NHWC format, channel comes last. 
    // if(this->isTest())
    // {
        std::string index_ssa =  "    %zero = constant 0.0 : f32 \n";
        index_ssa +=  "    %ci0 = constant 0 : index \n";
        index_ssa +=  "    %ci1 = constant 1 : index \n";
        std::unordered_map<int, int> input_index_map;         
        for (int i = 0; i < input_shape.size(); i++) 
        {
            // const auto [it]
            input_index_map.insert({input_shape[i], i});
        }

        for ( auto i : input_index_map)
        {
            index_ssa +=  "    %ci_Shape_" + std::to_string(i.first) + " = "+ " constant " + std::to_string(i.first) + " : index \n";
        }

        mlir << index_ssa << "\n";
    // }
    
    // Open Loop nest %a
    std::string loop_open = 
          std::string(4 + (num_loops-4) * 2, ' ')  
          + dict [FOR]
          + " %" + default_index_str[0] + " = %ci0 to "
          + "%ci_Shape_" +std::to_string(input_shape[0])
          + " step "+ "%ci" +"1\n"
          + std::string(4 + (num_loops-4) * 2, ' ') + "{\n"; 

    // Open Loop nest %d 
    loop_open += 
          std::string(4 + (num_loops-3) * 2, ' ')  
          + dict [FOR]
          + " %" + default_index_str[3] + " = %ci0 to "
          + "%ci_Shape_" + std::to_string(input_shape[3])
          + " step "+ "%ci" +"1\n"
          + std::string(4 + (num_loops-3) * 2, ' ') + "{"; 
    code = loop_open;
    mlir << code << "\n"; 

    // Alloc temporary variable to 2d vector ; global average pooling
    // auto buf2d_t0_name = cur_layer.getName() + "_buf2d_t0";
    // const auto [it_var2, flag2] = variable_map.insert({buf2d_t0_name,global_register_tracker++});
    // if(!flag2) {
    //     std::cout << "Variable map insertion failure" << std::endl;
    // }
    // auto buf2d_t0_reg = variable_map.at(buf2d_t0_name);
    // std::vector<unsigned> buf2d_t0_shape; 
    // buf2d_t0_shape.push_back(input_shape[1]);
    // buf2d_t0_shape.push_back(input_shape[2]);
    // auto buf2d_t0_memref = genMemRef(buf2d_t0_shape, cur_layer_dtype); // single element

    // auto output_reg = variable_map.at(cur_layer.getName());
    // auto output_memref = genMemRef(output_shape, cur_layer_dtype);
    // code = std::string(4 + (num_loops-4) * 2, ' ')
    //        + "    %" + std::to_string(buf2d_t0_reg) 
    //        + " = "
    //        + dict[ALLOC]
    //        + "() : "
    //        + buf2d_t0_memref;
    // mlir << code << "\n";

    // Temp Reduce Var 
    std::string temp_r_var = "%r2d_t0";

    // HxW dimension/size  -- integer
    std::string h_w = "\%h_w_dim";
    code = std::string(4 + (num_loops-3) * 2, ' ')
           + "    " + h_w
           + " = constant "
           + std::to_string(input_shape[1] * input_shape[2])
           + ".0 : "
        //    + " : i32 "
           + genDataType(cur_layer_dtype)
           + "\n";

    // Sum temp variable alloc 
    std::string temp_sum_var = "%sum2d_t0";
    std::vector<unsigned> temp_var_dim = {1};
    std::string tsv_memref = genMemRef(temp_var_dim, cur_layer_dtype);
    std::string temp_sum_load = "%sum2d_load";
    std::string temp_sum_store = "%sum2d_store";
    code += std::string(4 + (num_loops-3) * 2, ' ')
           + "    " + temp_sum_var
           + " = "
           + dict[ALLOC]
           + "() : "
           + genMemRef(temp_var_dim, cur_layer_dtype)
           + "\n";
    
    code += std::string(4 + (num_loops-3) * 2, ' ')
           + "    " 
           + dict[FILLVAR]
           + "(%zero,"
           + temp_sum_var
           + ") : f32, "
           + genMemRef(temp_var_dim, cur_layer_dtype) 
           + "\n";
    mlir << code;

    // Open Loop nest %b 
    loop_open = std::string(4 + (num_loops-1) * 2, ' ')  
           + dict [FOR]
           + " %" + default_index_str[1] + " = %ci0 to "
           + "%ci_Shape_" +std::to_string(input_shape[1])
           + " step "+ "%ci" +"1\n"
           + std::string(4 + (num_loops-1) * 2, ' ') + "{\n"; 

    // Open Loop nest %c 
    loop_open += 
          std::string(4 + (num_loops) * 2, ' ')  
          + dict [FOR]
          + " %" + default_index_str[2] + " = %ci0 to "
          + "%ci_Shape_" + std::to_string(input_shape[2])
          + " step "+ "%ci" +"1\n"
          + std::string(4 + (num_loops) * 2, ' ') + "{\n"; 
    code = loop_open;
    mlir << code; 

    // Load data to tmp var
    std::string tmpLoad = "%ltmp";
    code = std::string(4 + (num_loops+1)*2, ' ')
                + tmpLoad
                + " = " 
                + genLoad(default_index_str,
                          input_buffer_reg_0,
                          0,
                          input_shape.size(),
                          input_memref) + "\n";

    code += std::string(4 + (num_loops+1)*2, ' ')
                + temp_sum_load
                + " = " 
                + genLoad(default_index_str,
                          temp_sum_var,
                          0,
                          temp_var_dim.size(),
                          tsv_memref) + "\n";
    
    code += std::string(4 + (num_loops+1)*2, ' ')
            + temp_sum_store
            + " = "
            + dict[ADDF]
            + " "
            + temp_sum_load 
            + ", "
            + tmpLoad 
            + " : "
            + genDataType(cur_layer_dtype)
            + "\n";
    code += std::string(4 + (num_loops+1)*2, ' ')
            + genStore(default_index_str,
                        temp_sum_store,
                        temp_sum_var,
                        temp_var_dim, 
                        tsv_memref)
            + "\n";
    mlir << code;
    
    // Close loop %c and %d
    code = std::string(4 + (num_loops)*2, ' ') + "}\n";
    code += std::string(4 + (num_loops - 1)*2, ' ') + "}\n";
    mlir << code << "\n";

    code = std::string(4 + (num_loops - 2)*2, ' ')
            + temp_sum_load
            + " = " 
            + genLoad(default_index_str,
                      temp_sum_var,
                      0,
                      temp_var_dim.size(),
                      tsv_memref) + "\n";

    code += std::string(4 + (num_loops - 2)*2, ' ')
            + temp_r_var
            + " = "
            + dict[DIVF]
            + " "
            + temp_sum_load
            + ", "
            + h_w 
            + " : "
            + genDataType(cur_layer_dtype)
            + "\n";

    // Store data to buffer
    std::vector<unsigned> dflt_idx_vector_seq; 
    dflt_idx_vector_seq.push_back(0);
    dflt_idx_vector_seq.push_back(3);

    code += std::string(4 + (num_loops - 2)*2, ' ')
            + genStore(default_index_str,
                        temp_r_var,
                        output_reg,
                        dflt_idx_vector_seq, 
                        output_memref)
            + "\n";
    mlir << code << "\n";

    // Close loop %a and %d
    code = std::string(4 + (num_loops - 3)*2, ' ') + "}\n";
    code += std::string(4 + (num_loops - 4)*2, ' ') + "}\n";
    mlir << code << "\n";

}

void MLIRGen::genSoftMaxLayer(Layer& prev_layer,
                          Layer& cur_layer)
{

    mlir << "    // Layer type: Activation\n"
         << "    // Layre name: " << cur_layer.getName() << "\n";
    auto prev_layer_id = prev_layer.getID();
    auto cur_layer_id = cur_layer.getID();
    mlir << "    // Input from layer: " << prev_layer.getName() << "\n";
    auto input_buffer_reg = variable_map.at(prev_layer.getName());
    mlir << "    // Input buffer: %"
         << input_buffer_reg << " : ";
    auto& input_shape = prev_layer.getOutputDim();
    auto& input_dtype = prev_layer.getDataType();
    auto input_memref = genMemRef(input_shape, input_dtype);
    mlir << input_memref << "\n";

    if(this->isTest()) 
    {
        std::string code = "    %" + std::to_string(input_buffer_reg) 
                         + " = "
                         + dict[ALLOC]
                         + "() : "
                         + input_memref
                         + " \n";
        code += "    %c1 = constant 1.0 : f32 \n";                 
        code += "    " 
                + dict[FILLVAR]
                + "(%c1,%"
                + std::to_string(input_buffer_reg)
                + ") : f32, "
                + input_memref 
                + "\n";
        mlir << code << "\n";

    }

    // Output shape reminds the same
    mlir << "    // Output buffer: %"
         << input_buffer_reg << " : "
         << input_memref << "\n";
    cur_layer.setOutputDim(input_shape);

    std::string code;
    if (cur_layer.getActivation() == Layer::Activation::softmax)
    {
        mlir << "    // Activation: softmax\n";
        std::string cur_layer_dtype;
        if (cur_layer.getDataType() == Layer::Data_Type::f32)
        {
            cur_layer_dtype = "f32";
        }

        // Gen Exp
        auto temp_memref =  genMemRef(input_shape, input_dtype);
        code += "    // tmp buffer for exp eval \n";
        auto exp_reg_1_name = cur_layer.getName() + "_exp_reg_1";
        const auto [it_var1, flag1] = variable_map.insert({exp_reg_1_name,global_register_tracker++});
        if(!flag1) {
            std::cout << "Variable map insertion failure" << std::endl;
        }
        auto tmp_exp_reg = variable_map.at(exp_reg_1_name);
        code += "    \%" + std::to_string(tmp_exp_reg)
                + " = "
                + dict[ALLOC]
                + "() : "
                + temp_memref
                + "\n"
                ;

        auto temp_res_1_name = cur_layer.getName() + "_res_reg_1";
        const auto [it_var2, flag2] = variable_map.insert({temp_res_1_name,global_register_tracker++});
        if(!flag2) {
            std::cout << "Variable map insertion failure" << std::endl;
        }
        auto res_reg = variable_map.at(temp_res_1_name);
        code += "    // buffer for result of softmax (exp norm) \n";
        code += "    \%" + std::to_string(res_reg)
                + " = "
                + dict[ALLOC]
                + "() : "
                + temp_memref
                + "\n"
                ;
        // constant value
        // TODO: Global constant tracker?
        code += "    \%c0 = "
                + dict[CONSTANT]
                + " 0.0 :"
                + cur_layer_dtype
                + "\n"
                ;

        code += "    " + dict[FILLVAR]
                + "(\%"
                + std::to_string(tmp_exp_reg)
                + ", \%c0) : "
                + temp_memref
                + ", "
                + cur_layer_dtype
                + "\n"
                ;

        code += "    " + dict[FILLVAR]
                + "(\%"
                + std::to_string(res_reg)
                + ", \%c0) : "
                + temp_memref
                + ", "
                + cur_layer_dtype
                + "\n"
                ;
        // code += " /* -- InputBuffer : " + std::to_string(input_buffer_reg) + "*/ \n";
        code += genExp(input_buffer_reg, tmp_exp_reg
                    , input_shape, temp_memref, cur_layer_dtype);

        // Reduce Sum
        std::vector<unsigned> col_shape;
        col_shape.push_back(input_shape[1]);
        auto temp_memref2 = genMemRef(col_shape, input_dtype);
        auto temp_col_reg_1_name = cur_layer.getName() + "_col_reg_1";
        const auto [it_var3, flag3] = variable_map.insert({temp_col_reg_1_name,global_register_tracker++});
        if(!flag3) {
            std::cout << "Variable map insertion failure" << std::endl;
        }
        auto tmp_col_reg = variable_map.at(temp_col_reg_1_name);
        code += "    \%" + std::to_string(tmp_col_reg)
                + " = "
                + dict[ALLOC]
                + "() : "
                + temp_memref2
                + "\n"
                ;
        code += genNormReduceSum(res_reg, tmp_exp_reg, tmp_col_reg,
                            input_shape, temp_memref, temp_memref2, cur_layer_dtype);
    }
    mlir << code;
    mlir << "\n";
}

void MLIRGen::genEnd()
{
    mlir << "    return\n"
         << "}\n";
    mlir.close();
}

std::string MLIRGen::genMemRef(std::vector<unsigned> &dims,
                               Layer::Data_Type &d_type)
{
    std::string ret = "memref<";
    if(dims.size() > 0) {
        for (auto dim : dims)
        {
            ret += std::to_string(dim);
            ret += "x";
        }
    }

    switch (d_type) {
        case Layer::Data_Type::index :
            ret += "index>";
            break;
        case Layer::Data_Type::i32 :
            ret += "i32>";
            break;
        case Layer::Data_Type::f32 :
            ret += "f32>";
            break;
        default :
            // TODO: Proper exit handling
            exit (EXIT_FAILURE);
    }

    return ret;
}

std::string MLIRGen::genMemRef2(std::vector<unsigned> &dims,
                               Layer::Data_Type &d_type)
{
    std::string ret = "memref<";
    if(dims.size() > 0) {
        for (auto dim : dims)
        {
            ret += std::to_string(dim);
            ret += "x";
        }
    }

    switch (d_type) {
        case Layer::Data_Type::index :
            ret += "index";
            break;
        case Layer::Data_Type::i32 :
            ret += "i32";
            break;
        case Layer::Data_Type::f32 :
            ret += "f32";
            break;
        default :
            // TODO: Proper exit handling
            exit (EXIT_FAILURE);
    }

    return ret;
}

std::string MLIRGen::genTensor(std::vector<unsigned> &dims,
                               Layer::Data_Type &d_type)
{
    std::string ret = "tensor<";
    if(dims.size() > 0) {
        for (auto dim : dims)
        {
            ret += std::to_string(dim);
            ret += "x";
        }
    }

    switch (d_type) {
        case Layer::Data_Type::index :
            ret += "index>";
            break;
        case Layer::Data_Type::i32 :
            ret += "i32>";
            break;
        case Layer::Data_Type::f32 :
            ret += "f32>";
            break;
        default :
            // TODO: Proper exit handling
            exit (EXIT_FAILURE);
    }

    return ret;
}

std::string MLIRGen::genTensorConstF1D(std::vector<float> &vals,
                                       std::vector<unsigned> &dims,
                                       Layer::Data_Type &d_type)
{
    unsigned size_check = 1;
    for (auto dim : dims) size_check *= dim;
    assert(size_check);
    assert(dims.size() == 1);

    std::string ret = "";

    for (int m = 0; m < dims[0]; m++)
    {
        auto index = m;
        if (m == 0)
        {
            ret += ("[" + std::to_string(vals[index]) + ",");
        }
        // else if (m > 0 && m < dims[3] - 1)
        else if (m > 0 && m < dims[0] - 1)
        {
            ret += (std::to_string(vals[index]) + ",");
        }
        else
        {
            ret += (std::to_string(vals[index]) + "]");
        }
    }

    return ret;
}

std::string MLIRGen::genTensorConstF4D(std::vector<float> &vals,
                                       std::vector<unsigned> &dims,
                                       Layer::Data_Type &d_type)
{
    unsigned size_check = 1;
    for (auto dim : dims) size_check *= dim;
    assert(size_check);
    assert(dims.size() == 4);

    std::string ret = "";
    for (int i = 0; i < dims[0]; i++)
    {
        if (i == 0)
        {
            ret += "[";
        }
        for (int j = 0; j < dims[1]; j++)
        {
            if (j == 0)
            {
                ret += "[";
            }
            for (int k = 0; k < dims[2]; k++)
            {
                if (k == 0)
                {
                    ret += "[";
                }
                for (int m = 0; m < dims[3]; m++)
                {
                    auto index = i * dims[1] * dims[2] * dims[3] +
                                 j * dims[2] * dims[3] +
                                 k * dims[3] + m;
                    if (m == 0)
                    {
                        ret += ("[" + std::to_string(vals[index]) + ",");
                    }
		    else if (m > 0 && m < dims[3] - 1)
                    {
                        ret += (std::to_string(vals[index]) + ",");
                    }
                    else
                    {
                        ret += (std::to_string(vals[index]) + "]");
                    }
                }
                if (k < dims[2] - 1)
		{
                    ret += (",");
                }
                else
                {
                    ret += "]";
                }
            }
            if (j < dims[1] - 1)
            {
                ret += (",");
                // ret += (",\n");
            }
            else
            {
                ret += "]";
            }
        }
	if (i < dims[0] - 1)
        {
            // ret += (",\n\n");
            ret += (",");
        }
        else
        {
            ret += "]";
        }
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
    ret += "[";
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
    ret += "]";

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

    // -------------------------------------
    std::string index_ssa =  "    %zero_ = constant 0.0 : f32 \n";
    index_ssa +=  "    %ci0 = constant 0 : index \n";
    index_ssa +=  "    %ci1 = constant 1 : index \n";
    std::unordered_map<int, int> input_index_map;         
    for (int i = 0; i < shape.size(); i++) 
    {
        // const auto [it]
        input_index_map.insert({shape[i], i});
    }

    for ( auto i : input_index_map)
    {
        index_ssa +=  "    %ci_Shape_" + std::to_string(i.first) + " = "+ " constant " + std::to_string(i.first) + " : index \n";
    }

    mlir << index_ssa << "\n";

    // std::string code = "    %c0 = constant 0 : index \n";
    // code += "    %c1 = constant 1 : index \n";
    // mlir << code;

    // Gen loop statement
    for (auto i = 0; i < num_of_loops; i++)
    {
        std::string one_loop =
            std::string(4 + i * 2, ' ')
            + dict[FOR]
            + " %" + default_index_str[i] + " = %ci0 to "
            + "%ci_Shape_" + std::to_string(shape[i])
            + " step %ci1\n"
            + std::string(4 + i * 2, ' ') + "{\n";
        ret += one_loop;
    }

    // Gen loading
    auto load_str = std::string(4 + num_of_loops * 2, ' ')
                  + "%tmp = "
                  + genLoad(default_index_str,
                    // buffer_id, 0, num_of_loops - 1, shape_memref) + "\n";
                    buffer_id, 0, num_of_loops, shape_memref) + "\n";
                //   + genLoad(buffer_id, 0, num_of_loops - 1, shape_memref) + "\n";
                //   + genLoad(buffer_id, 0, num_of_loops, shape_memref) + "\n";
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
           "\%cond = " + dict[CMPLT] + ", %tmp, "+ zero+ " : " +
           dtype + "\n");

    // Gen if-store
    // Idx vector
    std::vector<unsigned> dflt_idx_vector_seq;
    for(int i = 0; i < num_of_loops; i++)
        dflt_idx_vector_seq.push_back(i);
    ret += (std::string(4 + num_of_loops * 2, ' ') 
            + dict[IF] + " \%cond\n" 
            + std::string(4 + num_of_loops * 2, ' ') 
            + "{\n" 
            + std::string(4 + (num_of_loops + 1) * 2, ' ') 
            + genStore(default_index_str,
                      zero, 
                      buffer_id, 
                      dflt_idx_vector_seq, 
                      shape_memref) 
            + "\n" 
            + std::string(4 + num_of_loops * 2, ' ') + "}\n");

    // Gen loop end
    for (auto i = num_of_loops - 1; i >= 0; i--)
    {
        std::string one_loop =
            std::string(4 + i * 2, ' ') + "}\n";
        ret += one_loop;
    }
    return ret;
}

// TODO, this function is not generic enough
// std::string MLIRGen::genExp(unsigned buffer_id,
//                              std::vector<unsigned> &shape,
//                              std::string &shape_memref,
//                              std::string &dtype)
// {

// }

// std::string MLIRGen::genSoftMax(unsigned buffer_id,
//                              std::vector<unsigned> &shape,
//                              std::string &shape_memref,
//                              std::string &dtype)
// {
//     std::string ret;
//     int num_of_loops = shape.size();
//     // auto &cur_layer_dtype = cur_layer.getDataType();

//     // Gen Exptmp
//     // auto temp_memref = genMemRef(shape, dtype);
//     // std::string code = "\%expt = "
//     //                 //  + " = "
//     //                  + dict[ALLOC]
//     //                  + "() : "
//     //                  + temp_memref
//     //                  + "\n"
//     //                  ;
//     // mlir << code;
//     mlir << "\n";
// }

std::string MLIRGen::genLoad(std::vector<std::string> &index_str,
                             unsigned buffer_id,
                             unsigned index_start,
                             unsigned index_end,
                             std::string& mem_ref)
{
    std::string ret = dict[LOAD] + " %" + std::to_string(buffer_id) + "[";
    for (int i = index_start; i < index_end; i++)
    {
        ret += ("%" + default_index_str[i]);
        if ((index_end - 1) > i)
            ret += (", ");
        else
            ret += ("] ");
    }
    // ret += ("%" + index_str[index_end] + "] : " + mem_ref);
    ret += (" : " + mem_ref);
    return ret;
}

std::string MLIRGen::genStore(std::vector<std::string> &index_str,
                              std::string &val,
                              unsigned buffer_id,
                              std::vector<unsigned> idx_vec_seq,
                              std::string& mem_ref)
{
    std::string ret = dict[STORE] + " " + val + ", "
                    + "%" + std::to_string(buffer_id) + "[";
    for (auto itr = idx_vec_seq.begin(); itr != idx_vec_seq.end(); itr++)
    {
        ret += ("%" + default_index_str[*itr]);
        if (itr < idx_vec_seq.end() - 1)
            ret += (", ");
    }
    ret += ("] : " + mem_ref);
    return ret;
}

std::string MLIRGen::genLoad(std::vector<std::string>& index_str,
                    std::string buffer_id,
                    unsigned index_start,
                    unsigned index_end,
                    std::string& mem_ref)
{
  std::string ret = dict[LOAD] + " " +buffer_id + "[";
  for (int i = index_start; i < index_end; i++)
  {
      ret += ("%" + default_index_str[i]);
      if ((index_end - 1) > i)
          ret += (", ");
      else
          ret += ("] ");
  }
  // ret += ("%" + index_str[index_end] + "] : " + mem_ref);
  ret += (" : " + mem_ref);
  return ret;
}

std::string MLIRGen::genStore(std::vector<std::string>& index_str,
                  std::string& val,
                  std::string buffer_id,
                  std::vector<unsigned> idx_vec_seq,
                  std::string& mem_ref)
{
  std::string ret = dict[STORE] + " " + val + ", "
                + buffer_id + "[";
  for (auto itr = idx_vec_seq.begin(); itr != idx_vec_seq.end(); itr++)
  {
      ret += ("%" + default_index_str[*itr]);
      if (itr < idx_vec_seq.end() - 1)
          ret += (", ");
  }
  ret += ("] : " + mem_ref);
  return ret;
}

std::string MLIRGen::genAdd(std::string& out_reg,
                            std::string& opr_1,
                            std::string& opr_2,
                            Layer::Data_Type& dtype)
{
    std::string opr, post_fix;

    switch (dtype)
    {
        case Layer::Data_Type::index :
            opr = dict[ADDI];
            post_fix = "index";
            break;
        case Layer::Data_Type::i32 :
            opr = dict[ADDI];
            post_fix = "i32";
            break;
        case Layer::Data_Type::f32 :
            opr = dict[ADDF];
            post_fix = "f32";
            break;
        default :
            // TODO: Proper exit handling
            exit (EXIT_FAILURE);
    }

    std::string ret = (out_reg + " = "
                    + opr + " "
                    + opr_1 + ", "
                    + opr_2 + " : " + post_fix);


    return ret;
}

std::string MLIRGen::genMult(std::string& out_reg,
                             std::string& opr_1,
                             std::string& opr_2,
                             Layer::Data_Type& dtype)
{
    std::string opr, post_fix;

    switch (dtype)
    {
        case Layer::Data_Type::index :
            opr = dict[MULI];
            post_fix = "index";
            break;
        case Layer::Data_Type::i32 :
            opr = dict[MULI];
            post_fix = "i32";
            break;
        case Layer::Data_Type::f32 :
            opr = dict[MULF];
            post_fix = "f32";
            break;
        default :
            // TODO: Proper exit handling
            exit (EXIT_FAILURE);
    }

    std::string ret = (out_reg + " = "
                    + opr + " "
                    + opr_1 + ", "
                    + opr_2 + " : " + post_fix);

    return ret;
}

std::string MLIRGen::genExp(unsigned buffer_id,
                            unsigned res_buffer_id,
                            std::vector<unsigned> &shape,
                            std::string &shape_memref,
                            std::string &dtype)
{
    std::string res;
    std::cout << " -- Res_buffer_id " << res_buffer_id << std::endl;
    // res += " /* -- InputBuffer : " + std::to_string(buffer_id) + "*/ \n";
    // Loop body begin:
    for (auto i = 0; i < shape.size(); ++i)
    {
        std::string loop_nest =
            std::string(4 + i * 2, ' ')
                + dict[FOR]
                + " %" + default_index_str[i] + " = 0 to "
                + std::to_string(shape[i])
                + " step 1\n"
                + std::string(4 + i * 2, ' ') + "{\n";
            res += loop_nest;
    }

    // Load val
    // TODO: (Vinay) Fix temp variable name/map
    std::string tmp_v_str = "%tmp";
    auto load_str = std::string(4 + shape.size() * 2, ' ')
                  + tmp_v_str+ " = "
                //   + genLoad(buffer_id, 0, shape.size() - 1, shape_memref) + "\n";
                  + genLoad(default_index_str,
                            buffer_id,
                            0,
                            shape.size(),
                            shape_memref) + "\n";
    res += load_str;
    // Eval exp
    std::string eval_v_str = "eval";
    auto exp_eval_str = std::string(4 + shape.size() * 2, ' ')
                  + "%" + eval_v_str + " = "
                  + dict[EXP]
                  + " %tmp : "
                  + dtype
                  + "\n"
                  ;
    res +=  exp_eval_str;
    // Store val
    // IDX vector
    std::vector<unsigned> dflt_idx_vector_seq;
    for(int i = 0; i < shape.size(); i++)
        dflt_idx_vector_seq.push_back(i);
    auto store_str = std::string(4 + shape.size() * 2, ' ')
                  + genStore(default_index_str,
                            eval_v_str,
                            res_buffer_id,
                            dflt_idx_vector_seq,
                            shape_memref )
                  + "\n";
    res += store_str;

    // Loop body end
    for (int i = shape.size() - 1; i >= 0; i--)
    {
        std::string loop_nest =
            std::string(4 + i * 2, ' ') + "}\n";
        res += loop_nest;
    }
    res += "\n";
    return res;
}

std::string MLIRGen::genNormReduceSum(unsigned res_buf,
                                      unsigned exp_buf,
                                      unsigned row_buf,
                                      std::vector<unsigned> &shape,
                                      std::string &shape2d_memref,
                                      std::string &shape1d_memref,
                                      std::string &dtype)
{
    std::string res = "";
    unsigned loop_lvl_cnt = 0;
    // std::cout << " -- " << index_str[loop_lvl_cnt] << std::endl;
    // std::cout << " -- " << dict[FOR] << std::endl;
    std::string outer_lp_begin =  std::string(4 + loop_lvl_cnt * 2, ' ')
                                   + dict[FOR]
                                   + " %" + default_index_str[loop_lvl_cnt] + " = 0 to "
                                   + std::to_string(shape[loop_lvl_cnt])
                                   + " step 1 {\n"
                                   ;
    loop_lvl_cnt += 1;
    // std::cout << " -- " <<  loop_nest_out_begin << std::endl;
    res += outer_lp_begin;
    std::vector<unsigned> shape_inner (1, shape[1]);
    std::string sum_ = "%sum";
    std::string sum_row = genReduceSum1D(exp_buf, row_buf, shape_inner, loop_lvl_cnt,
                            shape2d_memref, shape1d_memref, dtype, sum_);
    res += sum_row;

    std::string exp_norm = genExpNorm(res_buf, exp_buf, shape_inner, loop_lvl_cnt,
                            shape2d_memref, dtype, sum_);
    res += exp_norm;

    // close outer loop
    loop_lvl_cnt -= 1;
    std::string outer_lp_close = std::string(4 + loop_lvl_cnt * 2, ' ') + "}\n";
    res += outer_lp_close;

    return res;
}

std::string MLIRGen::genReduceSum1D(unsigned exp_buf,
                           unsigned row_buf,
                           std::vector<unsigned> &shape,
                           unsigned p_loop_lvl_cnt,
                           std::string &shape2d_memref,
                           std::string &shape1d_memref,
                           std::string &dtype,
                           std::string &sum_var)
{
    std::string res = "";
    // loop begin
    unsigned space_scaling = p_loop_lvl_cnt;
    for (auto i = 0; i < shape.size(); i++)
    {
        space_scaling += 1;
        std::string loop_nest =  std::string(4 +  space_scaling * 2, ' ')
                                + dict[FOR]
                                + " %" + default_index_str[p_loop_lvl_cnt] + " = 0 to "
                                + std::to_string(shape[i])
                                + " step 1 {\n"
                                ;
        res += loop_nest;
    }

    // Load unit data
    space_scaling += 1;
    std::string rstmp = "%rstmp";
    auto load_str = std::string(4 + space_scaling * 2, ' ')
                    + rstmp
                    + " = "
                    // TODO: Make it more generic
                    + genLoad(default_index_str,
                              exp_buf,
                              0,
                              1,
                              shape2d_memref)
                    + "\n";
    res += load_str;

    // Store data to temp row_vector
    // Idx vector
    std::vector<unsigned> dflt_idx_vector_seq;
    dflt_idx_vector_seq.push_back(1);
    auto store_str = std::string(4 + space_scaling * 2, ' ')
                    + genStore(default_index_str,
                               rstmp,
                               row_buf,
                                dflt_idx_vector_seq,
                               shape1d_memref)
                    + "\n"
                    ;
    res += store_str;

    // loop end
    for ( int i = shape.size() - 1; i >=0; i--)
    {
        space_scaling -= 1;
        std::string loop_nest = std::string(4 + space_scaling * 2, ' ') + "}\n";
        res += loop_nest;
    }

    // genSum1D -- Just sum result for 1D array
    // std::string sum_var = "%sum";
    std::string sum_init = "%sum_init";
    std::string sum_itr = "%sum_itr";
    std::string sum_new = "%sum_new";
    std::string stmp    = "%stmp";
    res += std::string(4 + space_scaling * 2, ' ')
           + sum_var
           + " = "
           + dict[FOR]
           // TODO: Fix hard coded values
           + " %" + default_index_str[0] + " = 0 to "
           + std::to_string(shape[0])
           + " step 1\n"
           ;

        space_scaling += 1;
        res += std::string(4 + space_scaling * 2, ' ')
               + "iter_args("
               + sum_itr
               + " = "
               + sum_init
               + ") -> "
               + dtype
               + " {\n"
               ;
        // load value for addition
        res += std::string(4 + space_scaling * 2, ' ')
                + stmp
                + " = "
                + genLoad(default_index_str,
                          row_buf,
                          0,
                          shape.size(),
                          shape1d_memref)
                + " \n"
                ;
        // addition
        res += std::string(4 + space_scaling * 2, ' ')
                + sum_new
                + " = "
                + dict[ADDF] + " "
                + sum_itr + ", "
                + stmp
                + " : "
                + dtype
                + "\n"
                ;
        res += std::string(4 + space_scaling * 2, ' ')
                + dict[YIELD] + " "
                + sum_new
                + " : "
                + dtype
                + "\n"
                ;

        // close sum scope
        space_scaling -= 1;
        res += std::string(4 + space_scaling * 2, ' ') + "}\n";

    return res;
}

std::string MLIRGen::genSum1D(std::string &sum_var, 
                              unsigned array_size, 
                              unsigned row_buf, 
                              std::string &row_shape_memref,
                              Layer::Data_Type &dtype,  
                              unsigned space_scaling) 
{
    // genSum1D -- Just sum result for 1D array
    // std::string sum_var = "%sum";
    std::string sum_init = "%sum_init";
    std::string sum_itr = "%sum_itr";
    std::string sum_new = "%sum_new";
    std::string stmp    = "%stmp";
    std::string res = std::string(4 + space_scaling * 2, ' ')
                       + sum_var
                       + " = "
                       + dict[FOR]
                       // TODO: Fix hard coded values
                       + " %" + default_index_str[0] + " = 0 to "
                       + std::to_string(array_size)
                       + " step 1\n"
                       ;

    space_scaling += 1;
    res += std::string(4 + space_scaling * 2, ' ')
           + "iter_args("
           + sum_itr
           + " = "
           + sum_init
           + ") -> "
           + genDataType(dtype)
           + " {\n"
           ;
    // load value for addition
    res += std::string(4 + space_scaling * 2, ' ')
            + stmp
            + " = "
            + genLoad(default_index_str,
                      row_buf,
                      0,
                      array_size,
                      row_shape_memref)
            + " \n"
            ;
    // addition
    res += std::string(4 + space_scaling * 2, ' ')
            + sum_new
            + " = "
            + dict[ADDF] + " "
            + sum_itr + ", "
            + stmp
            + " : "
            + genDataType(dtype)
            + "\n"
            ;
    res += std::string(4 + space_scaling * 2, ' ')
            + dict[YIELD] + " "
            + sum_new
            + " : "
            + genDataType(dtype)
            + "\n"
            ;

    // close sum scope
    space_scaling -= 1;
    res += std::string(4 + space_scaling * 2, ' ') + "}\n";

    return res; 
}

// std::string MLIRGen::genSum2D(std::string &sum_var, 
//                               std::vector<unsigned> buf_shape, 
//                               unsigned buf2d, 
//                               std::string &buf2d_shape_memref,
//                               Layer::Data_Type &dtype,  
//                               unsigned space_scaling) 
// {
//     // genSum1D -- Just sum result for 1D array
//     // std::string sum_var = "%sum";
//     std::string sum_init = "%sum_init";
//     std::string sum_itr = "%sum_itr";
//     std::string sum_new = "%sum_new";
//     std::string stmp    = "%stmp";
//     std::string res = std::string(4 + space_scaling * 2, ' ')
//                        + sum_var
//                        + " = "
//                        + dict[FOR]
//                        // TODO: Fix hard coded values
//                        + " %" + default_index_str[0] + " = 0 to "
//                        + std::to_string(buf_shape[0])
//                        + " step 1 {\n"
//                        ;

//     space_scaling += 1;
//     res = std::string(4 + space_scaling * 2, ' ')
//                        + sum_var
//                        + " = "
//                        + dict[FOR]
//                        // TODO: Fix hard coded values
//                        + " %" + default_index_str[1] + " = 0 to "
//                        + std::to_string(buf_shape[1])
//                        + " step 1 {\n"
//                        ;

//     // res += std::string(4 + space_scaling * 2, ' ')
//     //        + "iter_args("
//     //        + sum_itr
//     //        + " = "
//     //        + sum_init
//     //        + ") -> "
//     //        + genDataType(dtype)
//     //        + " {\n"
//     //        ;
//     space_scaling += 1;
//     // load value for addition
//     res += std::string(4 + space_scaling * 2, ' ')
//             + stmp
//             + " = "
//             + genLoad(default_index_str,
//                       buf2d,
//                       0,
//                       buf_shape[1],
//                       buf2d_shape_memref)
//             + " \n"
//             ;
//     // addition
//     res += std::string(4 + space_scaling * 2, ' ')
//             + sum_new
//             + " = "
//             + dict[ADDF] + " "
//             + sum_itr + ", "
//             + stmp
//             + " : "
//             + genDataType(dtype)
//             + "\n"
//             ;
//     res += std::string(4 + space_scaling * 2, ' ')
//             + dict[YIELD] + " "
//             + sum_new
//             + " : "
//             + genDataType(dtype)
//             + "\n"
//             ;

//     // close sum scope
//     space_scaling -= 1;
//     res += std::string(4 + space_scaling * 2, ' ') + "}\n";

//     return res; 
// }

std::string MLIRGen::genExpNorm(unsigned res_buf,
                               unsigned exp_buf,
                               std::vector<unsigned> &shape,
                               unsigned p_loop_lvl_cnt,
                               std::string &shape2d_memref,
                               std::string &dtype,
                               std::string &sum_var)
{
    std::string res = "";
    // std::cout << " -- LoopLevel: " << loop_lvl_cnt << std::endl;
    // loop begin
    unsigned space_scaling = p_loop_lvl_cnt;
    for (auto i = 0; i < shape.size(); i++)
    {
        space_scaling += 1;
        std::string loop_nest =  std::string(4 +  space_scaling * 2, ' ')
                                + dict[FOR]
                                + " %" + default_index_str[p_loop_lvl_cnt] + " = 0 to "
                                + std::to_string(shape[i])
                                + " step 1 {\n"
                                ;
        res += loop_nest;
    }

    // load variable
    space_scaling += 1;
    std::string tnorm1 = "%tnorm1";
    std::string load_exp = std::string(4 + space_scaling * 2, ' ')
                            + tnorm1
                            + " = "
                            // TODO: Make it more generic
                            + genLoad(default_index_str,
                                      exp_buf,
                                      0,
                                      2,
                                      shape2d_memref)
                            + "\n"
                            ;
    res += load_exp;

    // get norm value
    std::string tnorm2 = "%tnorm2";
    std::string norm_val = std::string(4 + space_scaling * 2, ' ')
                            + tnorm2
                            + " = "
                            + dict[DIVF] + " "
                            + tnorm1
                            + ", "
                            + sum_var
                            + " : "
                            + dtype
                            + "\n"
                            ;

    res += norm_val;
    // store value to result buf
    // Idx vector
    std::vector<unsigned> dflt_idx_vector_seq;
    for(int i = 0; i < 2; i++)
        dflt_idx_vector_seq.push_back(i);

    std::string store_res = std::string(4 + space_scaling * 2, ' ')
                            + genStore(default_index_str,
                                       tnorm2,
                                       res_buf,
                                       dflt_idx_vector_seq,
                                       shape2d_memref)
                            + " \n"
                            ;
    res += store_res;

    // close norm loop scope
    space_scaling -= 1;
    res += std::string(4 + space_scaling * 2, ' ') + "}\n";

    return res;
}

std::string MLIRGen::genDataType(Layer::Data_Type &d_type)
{
  std::string ret;
    switch (d_type) {
    case Layer::Data_Type::index :
        ret = "index";
        break;
    case Layer::Data_Type::i32 :
        ret = "i32";
        break;
    case Layer::Data_Type::f32 :
        ret = "f32";
        break;
    default :
        // TODO: Proper exit handling
        exit (EXIT_FAILURE);
    }
  return ret;
}

void MLIRGen::genPrintLayerId(unsigned id) 
{
    mlir << "    //Layer Id: " << id << "\n"; 

}

void MLIRGen::genPrintLayerName(Layer& cur_layer) 
{
    mlir << "    //Layer Name: " <<  cur_layer.getName() << "\n"; 

}

}
}
