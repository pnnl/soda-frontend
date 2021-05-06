#ifndef __MLIR_LINALG_GEN_HH__
#define __MLIR_LINALG_GEN_HH__

#include <fstream>
#include <map>

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
    std::map<std::string,int> variable_map; 
    // Track the output buffer ID of each layer
    
    // TODO: global array of constant values 
    // TODO: global tmp variable count; 
    std::vector<uint64_t> layer_output_buffer;

    // dictionary
    MLIRDict dict;

  protected:
    std::vector<std::string> default_index_str;

  public:
    MLIRGen(std::string &_fn) 
    { 
        default_index_str = 
            {"a", "b", "c", "d", "e", "f", "g", "h", "i",
             "j", "k", "l", "m", "n", "o", "p", "q", "r",
             "s", "t", "u", "v", "w", "x", "y", "z"};

        mlir.open(_fn); 
    }

    void genInit(std::vector<Layer> &layers);

    void genInputLayer(Layer& layer);
    void genConv2DLayer(Layer& prev_layer,
                        Layer& cur_layer);
    void genActLayer(Layer& prev_layer,
                     Layer& cur_layer);
    void genMaxPooling2DLayer(Layer& prev_layer,
                              Layer& cur_layer);
    void genFlattenLayer(Layer& prev_layer,
                              Layer& cur_layer);
    void genDenseLayer(Layer& prev_layer,
                              Layer& cur_layer);
    void genSoftMaxLayer(Layer& prev_layer,
                              Layer& cur_layer);

    void genEnd();

  protected:

    std::string genMemRef(std::vector<unsigned> &dims,
                          Layer::Data_Type &d_type);

    std::string genTensorConstF4D(std::vector<float> &vals,
                                  std::vector<unsigned> &dims,
                                  Layer::Data_Type &d_type);

    std::string genTensor(std::vector<unsigned> &dims,
                          Layer::Data_Type &d_type);

    std::string genDilations(std::vector<unsigned>&);

    std::string genPaddings(std::vector<std::vector<unsigned>>&);

    std::string genStrides(std::vector<unsigned>&);

    std::string genRelu(unsigned,
                        std::vector<unsigned>&,
                        std::string&,
                        std::string&);

    std::string genLoad(std::vector<std::string>&,
                        unsigned,
                        unsigned,
                        unsigned,
                        std::string&);

    std::string genStore(std::vector<std::string>&,
                         std::string&,
                         unsigned,
                         unsigned,
                         unsigned,
                         std::string&);

    std::string genAdd(std::string&,
                       std::string&,
                       std::string&,
                       Layer::Data_Type&);

    std::string genMult(std::string&,
                        std::string&,
                        std::string&,
                        Layer::Data_Type&);
    std::string genSoftMax(unsigned,
                          std::vector<unsigned>&,
                          std::string&,
                          std::string&);
    // std::string genLoad(unsigned,unsigned,unsigned,std::string&);

    // std::string genStore(std::string&,unsigned,unsigned,unsigned,std::string&);
    std::string genExp(unsigned,
                       unsigned,
                       std::vector<unsigned>&,
                       std::string&, 
                       std::string&);
    std::string genNormReduceSum(unsigned res_buf, 
                                 unsigned exp_buf, 
                                 unsigned row_buf,
                                 std::vector<unsigned> &shape, 
                                 std::string &shape2d_memref,
                                 std::string &shape1d_memref,
                                 std::string &dtype);
    std::string genReduceSum1D(unsigned exp_buf,
                               unsigned row_buf, 
                               std::vector<unsigned> &shape,
                               unsigned loop_lvl_cnt,
                               std::string &shape2d_memref,
                               std::string &shape1d_memref,
                               std::string &dtype,
                               std::string &sum_var);
    std::string genExpNorm(unsigned res_buf, 
                           unsigned exp_buf, 
                           std::vector<unsigned> &shape, 
                           unsigned loop_lvl_cnt,
                           std::string &shape2d_memref,
                           std::string &dtype,
                           std::string &sum_var);      
                                

    std::string genZeroF()
    {
        return "\%zero = constant 0.00000e+00 : f32";
    }
};
}
}

#endif
