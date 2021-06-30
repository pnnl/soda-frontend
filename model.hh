// Copyright 2021 Battelle Memorial Institute

#ifndef __MODEL_HH__
#define __MODEL_HH__

#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <utility>

#include "hdf5.h"

namespace SODA_FrontEnd
{
#define MAX_NAME 1024
class Model
{
  public:
    // TODO, capture more information from json file.
    class Layer
    {
      public:
        // TODO, more layer type:
        // https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215
	// https://medium.com/@zurister/depth-wise-convolution-and-depth-wise-separable-convolution-37346565d4ec
	// https://machinelearningmastery.com/introduction-to-1x1-convolutions-to-reduce-the-complexity-of-convolutional-neural-networks/

        // Batch-normalization: https://stackoverflow.com/questions/38553927/batch-normalization-in-convolutional-neural-network
        
        // TODO: Make layer type constant value. 
        enum class Layer_Type : int
        {
            Input, // the input layer
            Conv2D, // convolution layer
            Activation,
            BatchNormalization, // after convolution opr (conv, gamma, beta, mu, var, epsilon)
            Dropout,
            MaxPooling2D, // max polling layer
            AveragePooling2D,
            Flatten, // flatten layer
            Dense, // dense (fully-connected) layer
            ZeroPadding2D, // Pad zero values to the tensor HxW
            Add,  // Add convolution layers
            GlobalAveragePooling2D,
            MAX,
            DEFAULT
        }layer_type = Layer_Type::DEFAULT;
        auto getLayerType() { return layer_type; }

        enum class Data_Type : int
        {
            index,
            i32,
            f32,
        }d_type = Data_Type::f32;
        
        Layer() {}
        Layer(std::string &_name, Layer_Type &_type) : name(_name), layer_type(_type) {}
        
        std::vector<Layer *> inbound_layers; 
        unsigned id;
        void setID(unsigned _id) { id = _id; }
        auto getID() { return id; }

        std::string name; // Name of the layer
        auto &getName() { return name; }
        void insertInLayer(Layer * lyr) 
        {
            inbound_layers.push_back(lyr);
        }

        std::vector<Layer *> getInLayers() { return inbound_layers; }

        // TODO, tensor/memref consts, check, pre-initialization
        // weights/biases for CONV2D/Dense
        std::vector<unsigned> w_dims; // dims of the weights
        std::vector<float> weights; // all the weight
        std::vector<unsigned> b_dims; // dims of the bias
        std::vector<float> bias; // all the bias
        // dilation rage
        std::vector<unsigned> dilations;
        // Padding range
        std::vector<unsigned> paddings; 

        // Activation type
        enum class Activation : int
        {
            relu,
            softmax
        }activation = Activation::relu;

        // Padding type of the layer, used for CONV2D
        enum class Padding_Type : int
        {
            same,
            valid
        }padding_type = Padding_Type::valid;
        // strides, used for CONV2D/MaxPooling/AveragePooling
        std::vector<unsigned> strides;

        // TODO, need to extract more information
        // For batch-normalization
        std::vector<unsigned> beta_dims;
        std::vector<float> beta;
        std::vector<unsigned> gamma_dims;
        std::vector<float> gamma;
        std::vector<unsigned> moving_mean_dims;
        std::vector<float> moving_mean;
        std::vector<unsigned> moving_variance_dims;
        std::vector<float> moving_variance;

	    std::vector<unsigned> output_dims; // dimension of output

        void setDataType(std::string &_type)
        {
            if (_type == "float32") { d_type = Data_Type::f32; }
        }
        auto &getDataType() { return d_type; }

        void setActivation(std::string &_act)
        {
            if (_act == "relu") { activation = Activation::relu; }
            else if(_act == "softmax") {activation = Activation::softmax; }
        }
        auto &getActivation() { return activation; }

        void setWeights(std::vector<unsigned> &_w_dims,
                        std::vector<float> &_weights)
        {
            w_dims = _w_dims;
            weights = _weights;
        }
        auto &getKernel() { return weights; }
        auto &getKernelDim() { return w_dims; }
        void setBiases(std::vector<unsigned> &_b_dims,
                       std::vector<float> &_bias)
        {
            b_dims = _b_dims;
            bias = _bias;
        }
        void setStrides(std::vector<unsigned> &_strides)
        {
            strides = _strides;
        }
        auto &getStrides() { return strides; }
        void setDilations(std::vector<unsigned> &_dilations)
        {
            dilations = _dilations;
        }
        auto &getDilations() { return dilations; }
        void setPaddings(std::vector<unsigned> &_paddings)
        {
            paddings = _paddings;
        }
        auto &getPaddings() { return paddings;}
        void setBeta(std::vector<unsigned> &dims, std::vector<float> &data)
        {
            beta_dims = dims;
            beta = data;
        }
        void setGamma(std::vector<unsigned> &dims, std::vector<float> &data)
        {
            gamma_dims = dims;
            gamma = data;
        }
        void setMovingMean(std::vector<unsigned> &dims, std::vector<float> &data)
        {
            moving_mean_dims = dims;
            moving_mean = data;
        }
        void setMovingVariance(std::vector<unsigned> &dims, std::vector<float> &data)
        {
            moving_variance_dims = dims;
            moving_variance = data;
        }
        void setOutputDim(std::vector<unsigned> &_dims)
        {
            output_dims = _dims;
        }
        auto& getOutputDim() { return output_dims; }
        auto& getWeightDim() { return w_dims;}

        // TODO, new features for review
        double epsilon = 0.0;
        auto &getMovingMeanDim() { return moving_mean_dims;}
        auto &getMovingMeanData() { return moving_mean;}
        auto &getMovingVarianceDim() { return moving_variance_dims;}
        auto &getMovingVarianceData() { return moving_variance;}
        auto &getBeta() { return beta; }
        auto &getBetaDim() {return beta_dims;}
        auto &getGamma() { return gamma; }
        auto &getGammaDim() {return gamma_dims;}
        auto getEpsilon() { return epsilon; }
    };

    struct LayerMetaInfo 
    {
        LayerMetaInfo() {}
        LayerMetaInfo(unsigned lid, std::string lname, Model::Layer::Layer_Type ltype): layerId(lid), layer_name(lname), layer_type(ltype) {}
        unsigned layerId;
        std::string layer_name; 
        Model::Layer::Layer_Type layer_type = Model::Layer::Layer_Type::DEFAULT;
    };

    // Model - Architecture - Layer Generation
    class TestLayer 
    {
      public:
        TestLayer() {}
        TestLayer(std::vector<Model::Layer>& layers_) {layers = layers_;}

        
        // void getInputDim (Model::Layer& layer) {}
        // void getOutputDim (Model::Layer& layer) {}

        void initializeLayers(std::vector<Model::Layer>& layers_) {layers = layers_;} 
        std::size_t numLayers() { return layers.size();}
        
        // Bin each layer to the respective Layer_type using the map operator.
        void binLayers();
      private: 
        std::vector<Model::Layer> layers;
        std::unordered_map<Layer::Layer_Type, std::vector<LayerMetaInfo>> layer_meta_info;
        // std::vector<LayerTypeInfo> layer_type_info;
        std::vector<unsigned> input_dim;
        std::vector<unsigned> output_dim;
        // TODO: Keep a list of different layer types in different buckets
    };

    // Model - Architecture
    class Architecture
    {
      protected:
        std::vector<Layer> layers;

        std::unordered_map<std::string, std::vector<std::string>>
            layer_inputs;

        std::string mlir_gen_fn;
      public:
        Architecture() {}

        void setMLIRGeneratorFile(std::string _fn)
        {
            // TODO, set it to code-gen class
            mlir_gen_fn = _fn;
        }

        void addLayer(std::string &_name, Layer::Layer_Type &_type)
        {
            layers.emplace_back(_name, _type);
        }

        // TODO, better to use unordered map here
        Layer& getLayer(std::string &_name)
        {
            for (auto &layer : layers)
            {
                if (layer.name == _name) { return layer; }
            }
            std::cout << _name << "\n";
            std::cerr << "Error: layer is not found.\n";
            exit(0);
            
        }
       
        void trackLayerConnection(std::string &cur_layer,
                                  std::vector<std::string> &inbounds)
        {
            auto map_iter = layer_inputs.find(cur_layer);
            assert(map_iter == layer_inputs.end());
            layer_inputs.insert({cur_layer, inbounds});
        }

        auto &getLayers() { return layers; }

        void MLIRGenerator();

        void printLayers() // Only used for small network debuggings.
        {
            for (auto &layer : layers)
            {
                auto name = layer.name;
                auto type = layer.layer_type;

                std::cout << "Layer name: " << name << "; ";
                if (type == Layer::Layer_Type::Input) 
                { 
                    std::cout << "Layer type: Input" << std::endl;
                    auto &output_dims = layer.getOutputDim();
                    std::cout << "Input Shape: ";
                    for (auto dim : output_dims) { std::cout << dim << " "; }
                    std::cout << std::endl; 

                    std::cout << "Output Dims: ";
                    // auto &output_dims = layer.getOutputDim();
                    for (auto dim : output_dims) { std::cout << dim << " "; }
                    std::cout << std::endl; 
                    
                }
                else if (type == Layer::Layer_Type::Conv2D) 
                { 
                    std::cout << "Layer type: Conv2D" << std::endl;
                    std::cout << "Output Dims: ";
                    auto &output_dims = layer.getOutputDim();
                    for (auto dim : output_dims) { std::cout << dim << " "; }
                    std::cout << std::endl; 
                }
                else if (type == Layer::Layer_Type::Activation) 
                { 
                    std::cout << "Layer type: Activation";
                    if(layer.activation == Layer::Activation::relu)
                        std::cout << ", Activation Type: relu" << std::endl;
                    else if(layer.activation == Layer::Activation::softmax)
                        std::cout << ", Activation Type: softmax" << std::endl;
                    else 
                        std::cout << std::endl;
                    std::cout << "Output Dims: ";
                    auto &output_dims = layer.getOutputDim();
                    for (auto dim : output_dims) { std::cout << dim << " "; }
                    std::cout << std::endl;
                }
                else if (type == Layer::Layer_Type::BatchNormalization) 
                { 
                    std::cout << "Layer type: BatchNormalization" << std::endl; 
                    std::cout << "Output Dims: ";
                    auto &output_dims = layer.getOutputDim();
                    for (auto dim : output_dims) { std::cout << dim << " "; }
                    std::cout << std::endl;
                }
                else if (type == Layer::Layer_Type::Dropout) 
                { 
                    std::cout << "Layer type: Dropout" << std::endl; 
                    std::cout << "Output Dims: ";
                    auto &output_dims = layer.getOutputDim();
                    for (auto dim : output_dims) { std::cout << dim << " "; }
                    std::cout << std::endl;
                }
                else if (type == Layer::Layer_Type::MaxPooling2D) 
                { 
                    std::cout << "Layer type: MaxPooling2D" << std::endl; 
                    std::cout << "Output Dims: ";
                    auto &output_dims = layer.getOutputDim();
                    for (auto dim : output_dims) { std::cout << dim << " "; }
                    std::cout << std::endl;
                }
                else if (type == Layer::Layer_Type::AveragePooling2D) 
                { 
                    std::cout << "Layer type: AveragePooling2D" << std::endl; 
                    std::cout << "Output Dims: ";
                    auto &output_dims = layer.getOutputDim();
                    for (auto dim : output_dims) { std::cout << dim << " "; }
                    std::cout << std::endl;
                }
                else if (type == Layer::Layer_Type::Flatten) 
                { 
                    std::cout << "Layer type: Flatten" << std::endl; 
                    std::cout << "Output Dims: ";
                    auto &output_dims = layer.getOutputDim();
                    for (auto dim : output_dims) { std::cout << dim << " "; }
                    std::cout << std::endl;
                }
                else if (type == Layer::Layer_Type::Dense)
                { 
                    std::cout << "Layer type: Dense" << std::endl; 
                    std::cout << "Output Dims: ";
                    auto &output_dims = layer.getOutputDim();
                    for (auto dim : output_dims) { std::cout << dim << " "; }
                    std::cout << std::endl;
                }
                else if (type == Layer::Layer_Type::ZeroPadding2D)
                {
                    std::cout << "Layer type: ZeroPadding2D" << std::endl; 
                    std::cout << "Output Dims: ";
                    auto &output_dims = layer.getOutputDim();
                    for (auto dim : output_dims) { std::cout << dim << " "; }
                    std::cout << std::endl;
                }
                else if (type == Layer::Layer_Type::Add)
                {
                    std::cout << "Layer type: Add" << std::endl; 
                    std::cout << "Output Dims: ";
                    auto &output_dims = layer.getOutputDim();
                    for (auto dim : output_dims) { std::cout << dim << " "; }
                    std::cout << std::endl;
                }
                else if (type == Layer::Layer_Type::GlobalAveragePooling2D)
                {
                    std::cout << "Layer type: GlobalAveragePooling2D" << std::endl; 
                    std::cout << "Output Dims: ";
                    auto &output_dims = layer.getOutputDim();
                    for (auto dim : output_dims) { std::cout << dim << " "; }
                    std::cout << std::endl;
                }
                else { std::cerr << "Error: unsupported layer type\n"; exit(0); }
                std::cout << "\n";
              
                // if (type == Layer::Layer_Type::Input)
                // {
                //     auto &output_dims = layer.output_dims;
                //     std::cout << "Input shape: ";
                //     for (auto dim : output_dims) { std::cout << dim << " "; }
                //     std::cout << "\n";
                // }

                auto &w_dims = layer.w_dims;
                auto &weights = layer.weights;
                auto &b_dims = layer.b_dims;
                auto &bias = layer.bias;
                if (weights.size())
                {
                    std::cout << "Weights dim (" << weights.size() << "): ";
                    for (auto dim : w_dims) { std::cout << dim << " "; }
                    std::cout << "\n";
                }

                if (bias.size())
                {
                    std::cout << "Bias dim (" << bias.size() << "): ";
                    for (auto dim : b_dims) { std::cout << dim << " "; }
                    std::cout << "\n";
                }

                std::cout << "Total params: " << weights.size() + bias.size() << "\n";

                std::cout << "\n";
            }
        }
    };

    

    Architecture arch;

  public:
    Model(std::string &arch_file, 
          std::string &weight_file,
          std::string &mlir_gen)
    {
        loadArch(arch_file);
        loadWeights(weight_file);

        arch.setMLIRGeneratorFile(mlir_gen);
    }

    void printLayers() { arch.printLayers(); }

    void MLIRGenerator() { arch.MLIRGenerator(); }

  protected:
    void loadArch(std::string &arch_file);
    void loadWeights(std::string &weight_file);

  protected:
    void scanGroup(hid_t);
    void extrWeights(hid_t);
};
}

#endif
