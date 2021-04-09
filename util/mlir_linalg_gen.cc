#include "mlir_linalg_gen.hh"

#include <cassert>
#include <cmath>
#include <cassert>
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
    mlir << "    // Layer type: Conv2D\n"
         << "    // Layer name: " << cur_layer.getName() << "\n";
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
        // No padding
        out_height = ceil(float(input_shape[1] - kernel[0]) / 
                     float(stride[0])) + 1;
	
        out_width = ceil(float(input_shape[2] - kernel[1]) / 
                     float(stride[1])) + 1;
    }
    else if (cur_layer.padding_type == Layer::Padding_Type::same)
    {
        // Padding Present
        // TODO: consider other cases
        // Kernel size normally odd. 
        unsigned padding_size = int(kernel[0])/2; 
        // out_height = ceil(float(input_shape[1]) / float(stride[0]));
	
        // out_width = ceil(float(input_shape[2]) / float(stride[1]));

        out_height = ceil(float(input_shape[1] + 2 * padding_size - kernel[0]) / 
                     float(stride[0])) + 1;
	
        out_width = ceil(float(input_shape[2] + 2 * padding_size - kernel[1]) / 
                     float(stride[1])) + 1;

        // unsigned pad_along_height = std::max(int((out_height - 1) * stride[0] 
        //                           + kernel[0] - input_shape[1]), 0);

        // unsigned pad_along_width = std::max(int((out_width - 1) * stride[1] 
        //                          + kernel[1] - input_shape[2]), 0);
        // std::cout << pad_along_height << " " << pad_along_width << "\n";
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
    mlir << "    // Layer type: Activation\n"
         << "    // Layer name: " << cur_layer.getName() << "\n";
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
    mlir << "    // Layer type: MaxPooling2D\n"
         << "    // Layer name: " << cur_layer.getName() << "\n";
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

void MLIRGen::genFlattenLayer(Layer& prev_layer,
                              Layer& cur_layer)
{
    mlir << "    // Layer type: Flatten\n"
         << "    // Layer name: " << cur_layer.getName() << "\n";
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

    // Determine output size
    auto &cur_layer_dtype = cur_layer.getDataType();
    unsigned out_size = 1;
    for (auto dim : input_shape) { out_size *= dim; }
    mlir << "    // Output size: " << out_size << "\n";
    std::vector<unsigned> out_dim = {out_size};
    layer_output_buffer.push_back(global_register_tracker);
    cur_layer.setOutputDim(out_dim);

    auto out_buffer_reg = global_register_tracker++;
    std::string out_memref = genMemRef(out_dim, cur_layer_dtype);
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
            std::to_string(out_buffer_reg) + "[\%index] : " + 
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
    auto input_buffer_reg = layer_output_buffer[prev_layer_id];
    mlir << "    // Input buffer: %"
         << input_buffer_reg << " : ";
    auto& input_shape = prev_layer.getOutputDim();
    auto& input_dtype = prev_layer.getDataType();
    auto input_memref = genMemRef(input_shape, input_dtype);
    mlir << input_memref << "\n";


    // Determine output and kernel dimension
    auto &kernel_dim = cur_layer.getKernelDim();
    assert(input_shape.size() == 1); // Input must be 1D
    mlir << "    // Kernel dim.: ";
    // weight_dim[0] - input dimension
    // weight_dim[1] - output dimension
    for (auto dim : kernel_dim) { mlir << dim << " "; }
    mlir << "\n";
    mlir << "    // Output size: " << kernel_dim[1] << "\n";

    auto &cur_layer_dtype = cur_layer.getDataType();
    auto kernel_reg = global_register_tracker;
    auto kernel_memref = genMemRef(kernel_dim, cur_layer_dtype);
    std::string code = "    %" + std::to_string(kernel_reg)
                     + " = "
                     + dict[ALLOC]
                     + "() : "
                     + kernel_memref;
    mlir << code << "\n";
    global_register_tracker++;

    // alloc output
    std::vector<unsigned> output_shape = {kernel_dim[1]};
    auto output_reg = global_register_tracker;
    auto output_memref = genMemRef(output_shape, cur_layer_dtype);
    code = "    %" + std::to_string(output_reg)
                   + " = "
                   + dict[ALLOC]
                   + "() : "
                   + output_memref;
    mlir << code << "\n\n";
    layer_output_buffer.push_back(global_register_tracker);
    cur_layer.setOutputDim(output_shape);
    global_register_tracker++;

    // Dense function
    code = "";
    int num_of_loops = kernel_dim.size();
    // Gen loop statement
    for (auto i = 0; i < num_of_loops; i++)
    {
        auto index = num_of_loops - 1 - i;

        std::string one_loop = 
            std::string(4 + i * 2, ' ')
            + dict[FOR] 
            + " %" + default_index_str[i] + " = 0 to " 
            + std::to_string(kernel_dim[index])
            + " step 1\n"
            + std::string(4 + i * 2, ' ') + "{\n";

        code += one_loop; 
        if (i == 0)
        {
            code += (std::string(4 + (i + 1) * 2, ' ')
                 + "\%out_val = " 
                 + dict[ADDI] + " \%zero, \%zero : i32\n");
        }
    } 

    // Gen load
    std::vector<std::string> input_load_index;
    std::vector<std::string> kernel_load_index;
    std::vector<std::string> output_store_index;
    for (auto i = 0; i < num_of_loops; i++)
    {
        auto index = num_of_loops - 1 - i;
        if (i == 0) 
        {
            input_load_index.push_back(default_index_str[index]);
        }

        if (i == (num_of_loops -1))
        {
            output_store_index.push_back(default_index_str[index]);
        }

        kernel_load_index.push_back(default_index_str[index]);
    }
    auto load_str = std::string(4 + num_of_loops * 2, ' ')
                  + "\%w_val = "
                  + genLoad(kernel_load_index,
                            kernel_reg, 
                            0,
                            num_of_loops - 1,
                            kernel_memref) + "\n";
    code += load_str;
    load_str = std::string(4 + num_of_loops * 2, ' ')
             + "\%in_val = "
             + genLoad(input_load_index,
                       input_buffer_reg, 
                       0,
                       0,
                       input_memref) + "\n";
    code += load_str;

    // Gen multiply and accumulate
    std::string dest = "\%out_tmp";
    std::string opr_1 = "\%in_val";
    std::string opr_2 = "\%w_val";
    auto mult_str = std::string(4 + num_of_loops * 2, ' ')
                  + genMult(dest, opr_1, opr_2, 
                            cur_layer.getDataType())
                  + "\n";
    code += mult_str;

    dest = "\%out_val";
    opr_1 = "\%out_val";
    opr_2 = "\%out_tmp";
    auto add_str = std::string(4 + num_of_loops * 2, ' ')
                 + genAdd(dest, opr_1, opr_2, 
                           cur_layer.getDataType())
                 + "\n";
    code += add_str;

    // Gen loop end
    for (auto i = num_of_loops - 1; i >= 0; i--)
    {
        std::string one_loop = 
            std::string(4 + i * 2, ' ') + "}\n";
        code += one_loop; 

        if (i == (num_of_loops - 1))
        {
            std::string to_be_stored = "\%out_val";
            code += (std::string(4 + i * 2, ' ') +
                     genStore(output_store_index,
                     to_be_stored,
                     output_reg, 0, 0, output_memref) + "\n");
        }
    }

    mlir << code;

    mlir << "\n";
}

void MLIRGen::genSoftMaxLayer(Layer& prev_layer,
                          Layer& cur_layer)
{

    mlir << "    // Layer type: Activation\n"
         << "    // Layre name: " << cur_layer.getName() << "\n";
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

    // layer_output_buffer.push_back(input_buffer_reg);
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
        auto tmp_exp_reg = global_register_tracker;
        code += "    \%" + std::to_string(tmp_exp_reg)
                + " = "
                + dict[ALLOC]
                + "() : "
                + temp_memref
                + "\n"
                ;
        global_register_tracker++;

        auto res_reg = global_register_tracker;
        code += "    // buffer for result of softmax (exp norm) \n";
        code += "    \%" + std::to_string(res_reg)
                + " = "
                + dict[ALLOC]
                + "() : "
                + temp_memref
                + "\n"
                ;
        global_register_tracker++;

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
        // std::cout << " -- TEST 1" << std::endl;
        auto temp_memref2 = genMemRef(col_shape, input_dtype);
        // std::cout << " -- TEST 2" << std::endl;
        auto tmp_col_reg = global_register_tracker;
        code += "    \%" + std::to_string(tmp_col_reg)
                + " = "
                + dict[ALLOC]
                + "() : "
                + temp_memref2
                + "\n"
                ;
        global_register_tracker++;
        // std::cout << " -- TEST 3" << std::endl;
        code += genNormReduceSum(res_reg, tmp_exp_reg, tmp_col_reg, 
                            input_shape, temp_memref, temp_memref2, cur_layer_dtype);
        // std::cout << " -- TEST 4" << std::endl;
    }
    mlir << code;
    mlir << "\n";

    // std::cout << " -- Layer Name: " << cur_layer.name << std::endl;
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
            + " %" + default_index_str[i] + " = 0 to " 
            + std::to_string(shape[i])
            + " step 1\n"
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
           "\%cond = " + dict[CMPLT] + ", %tmp, \%zero : " + 
           dtype + "\n");

    // Gen if-store
    ret += (std::string(4 + num_of_loops * 2, ' ') + 
            dict[IF] + " \%cond\n" +
            std::string(4 + num_of_loops * 2, ' ') + "{\n" +
            std::string(4 + (num_of_loops + 1) * 2, ' ') +
            genStore(default_index_str,
                // zero, buffer_id, 0, num_of_loops - 1, shape_memref) + "\n" +
                zero, buffer_id, 0, num_of_loops, shape_memref) + "\n" +
            // genStore(zero, buffer_id, 0, num_of_loops - 1, shape_memref) + "\n" +
            // genStore(zero, buffer_id, 0, num_of_loops, shape_memref) + "\n" +
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

// TODO, this function is not generic enough
// std::string MLIRGen::genExp(unsigned buffer_id,
//                              std::vector<unsigned> &shape,
//                              std::string &shape_memref,
//                              std::string &dtype)
// {

// }

std::string MLIRGen::genSoftMax(unsigned buffer_id,
                             std::vector<unsigned> &shape,
                             std::string &shape_memref,
                             std::string &dtype)
{
    std::string ret; 
    int num_of_loops = shape.size();
    // auto &cur_layer_dtype = cur_layer.getDataType();

    // Gen Exptmp
    // auto temp_memref = genMemRef(shape, dtype);
    // std::string code = "\%expt = "
    //                 //  + " = "
    //                  + dict[ALLOC]
    //                  + "() : "
    //                  + temp_memref 
    //                  + "\n"
    //                  ;
    // mlir << code; 
    mlir << "\n";
}

std::string MLIRGen::genLoad(std::vector<std::string> &index_str,
                             unsigned buffer_id,
// std::string MLIRGen::genLoad(unsigned buffer_id,
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

// TODO, this function is not generic enough
std::string MLIRGen::genStore(std::vector<std::string> &index_str,
                              std::string &val,
                              unsigned buffer_id,
                              unsigned index_start,
                              unsigned index_end,
                              std::string& mem_ref)
{
    std::string ret = dict[STORE] + " " + val + ", "
                    + "%" + std::to_string(buffer_id) + "[";
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
    auto store_str = std::string(4 + shape.size() * 2, ' ') 
                //   + genStore(eval_v_str, res_buffer_id, 0, shape.size() - 1, shape_memref ) 
                  + genStore(default_index_str, 
                            eval_v_str, 
                            res_buffer_id, 
                            0, 
                            shape.size(), 
                            shape_memref ) 
                  + "\n";
    res += store_str; 

    // res += " /* -- Shape.size : " + std::to_string(shape.size()) + "*/ \n";
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
    auto store_str = std::string(4 + space_scaling * 2, ' ')
                    // TODO: Fix this hard coded part. 
                    // + genStore(rstmp, row_buf, 1, shape.size(), shape1d_memref)
                    + genStore(default_index_str, 
                               rstmp, 
                               row_buf, 
                               1, 
                               2, 
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
    std::string store_res = std::string(4 + space_scaling * 2, ' ')
                            + genStore(default_index_str,
                                       tnorm2, 
                                       res_buf, 
                                       0, 
                                       2, 
                                       shape2d_memref)
                            + " \n"
                            ; 
    res += store_res;
    
    // close norm loop scope
    space_scaling -= 1; 
    res += std::string(4 + space_scaling * 2, ' ') + "}\n";

    return res; 
}                              

}
}

